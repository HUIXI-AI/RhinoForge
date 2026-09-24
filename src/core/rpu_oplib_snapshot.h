// One immutable Linux file image for every Program loaded from an op library.
#pragma once

#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <linux/memfd.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

class OplibSnapshot {
    struct Fd {
        int value = -1;
        ~Fd() { if (value >= 0) ::close(value); }
        Fd() = default;
        Fd(const Fd&) = delete;
        Fd& operator=(const Fd&) = delete;
    } fd_;
    size_t size_ = 0;
    static constexpr int kSeals =
        F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;

    static void require(bool valid, const char* detail) {
        if (!valid) throw std::runtime_error(std::string("oplib snapshot: ") + detail);
    }

public:
    explicit OplibSnapshot(const std::string& source) {
        Fd input;
        input.value = ::open(source.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
        require(input.value >= 0, "cannot open canonical source");
        struct stat before{}, after{};
        // ponytail: current refs are ~17 MiB and 276 KiB. Reject >256 MiB
        // explicitly; never fall back to loading a mutable source path.
        require(::fstat(input.value, &before) == 0 && S_ISREG(before.st_mode) &&
                    before.st_size > 0 && before.st_size <= 256 * 1024 * 1024,
                "source must be a regular nonempty file at most 256 MiB");
        fd_.value = static_cast<int>(::syscall(
            SYS_memfd_create, "rpu-oplib", MFD_CLOEXEC | MFD_ALLOW_SEALING));
        require(fd_.value >= 0, "memfd_create failed");
        size_ = static_cast<size_t>(before.st_size);
        char buffer[65536];
        size_t offset = 0;
        while (offset < size_) {
            const size_t wanted = std::min(sizeof(buffer), size_ - offset);
            const ssize_t count = ::pread(input.value, buffer, wanted, offset);
            if (count < 0 && errno == EINTR) continue;
            require(count > 0, "incomplete source read");
            size_t written = 0;
            while (written < static_cast<size_t>(count)) {
                const ssize_t n = ::write(fd_.value, buffer + written, count - written);
                if (n < 0 && errno == EINTR) continue;
                require(n > 0, "incomplete snapshot write");
                written += static_cast<size_t>(n);
            }
            offset += static_cast<size_t>(count);
        }
        require(::fstat(input.value, &after) == 0 &&
                    before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
                    before.st_size == after.st_size &&
                    before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
                    before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
                    before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
                    before.st_ctim.tv_nsec == after.st_ctim.tv_nsec,
                "source changed during snapshot read");
        require(::fcntl(fd_.value, F_ADD_SEALS, kSeals) == 0 &&
                    ::fcntl(fd_.value, F_GET_SEALS) == kSeals,
                "cannot seal snapshot");
    }

    std::string path() const { return "/proc/self/fd/" + std::to_string(fd_.value); }

    // Read only for the native payload-identity hash; the bytes remain opaque.
    // Never reopen the mutable source path, even after a successful load.
    std::string bytes() const {
        require(::fcntl(fd_.value, F_GET_SEALS) == kSeals, "snapshot seals changed");
        std::string result(size_, '\0');
        size_t offset = 0;
        while (offset < size_) {
            const ssize_t n = ::pread(fd_.value, result.data() + offset, size_ - offset, offset);
            if (n < 0 && errno == EINTR) continue;
            require(n > 0, "incomplete sealed snapshot read");
            offset += static_cast<size_t>(n);
        }
        return result;
    }
};
