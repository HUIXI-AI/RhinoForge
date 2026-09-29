"""Run production mask preparation with observable CPU/device copy boundaries."""
from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition


def test_mask_goes_directly_to_its_stable_slot_with_padding_and_ordinals(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (Path(__file__).parents[1] / "src/ops/rpu_sdpa.cpp").read_text()
    method = _definition(source, "PreparedMask sdpa_prepare_mask(")
    program = r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
int publications=0, downloads=0;
namespace c10 { template<class T> using optional=std::optional<T>; enum class DeviceType { CPU=0, RPU=1 }; }
namespace at {
constexpr int kCPU=0,kPrivateUse1=1,kHalf=2,kFloat=3;
struct Device { int value; c10::DeviceType type() const {return static_cast<c10::DeviceType>(value);} };
struct Options { int dev=0,dtype=kHalf; Options device(int d) const {return {d,dtype};} };
struct Tensor {
    std::vector<int64_t> shape;
    std::shared_ptr<std::vector<float>> data;
    Options opts;
    Tensor()=default;
    Tensor(std::vector<int64_t> s,Options o={}):shape(s),opts(o) {
        int64_t count=1; for(auto n:s)count*=n; data=std::make_shared<std::vector<float>>(count,0.f);
    }
    bool defined() const {return bool(data);}
    int dim() const {return shape.size();}
    int64_t size(int i) const {return shape.at(i);}
    int scalar_type() const {return opts.dtype;}
    Device device() const {return {opts.dev};}
    Options options() const {return opts;}
    Tensor select(int axis,int index) const {
        assert(axis==0 && index==0 && shape[0]==1); auto t=*this; t.shape.erase(t.shape.begin());return t;
    }
    Tensor contiguous() const {return *this;}
    Tensor to(int kind) const {
        auto t=*this;
        if(kind==kPrivateUse1)throw std::runtime_error("temporary RPU upload");
        if(kind==kCPU) {if(opts.dev==kPrivateUse1)++downloads; t.opts.dev=kCPU;}
        else t.opts.dtype=kind;
        return t;
    }
    void copy_(const Tensor& src) {
        assert(opts.dev==kPrivateUse1 && src.opts.dev==kCPU && shape==src.shape);
        *data=*src.data; ++publications;
    }
};
Tensor constant_pad_nd(const Tensor& t,std::initializer_list<int64_t> pad,float value) {
    auto n=*(pad.begin()+1);Tensor out({t.size(0),t.size(1)+n},t.opts);
    for(int r=0;r<t.size(0);++r)for(int c=0;c<out.size(1);++c)
        (*out.data)[r*out.size(1)+c]=c<t.size(1)?(*t.data)[r*t.size(1)+c]:value;
    return out;
}
}
int64_t Align(int64_t n,int64_t a){return (n+a-1)/a*a;}
struct PreparedMask {at::Tensor ddr_tensor;int mask_type=0;};
struct SdpaStableMaskCache {
    std::map<std::tuple<int64_t,int64_t,int64_t>,at::Tensor> slots;
    at::Tensor stable_slot(int64_t q,int64_t k,at::Options opts,int64_t ordinal) {
        assert(opts.dev==at::kPrivateUse1 && opts.dtype==at::kHalf);
        auto key=std::make_tuple(q,k,ordinal);auto it=slots.find(key);
        if(it==slots.end())it=slots.emplace(key,at::Tensor({q,k},opts)).first;
        return it->second;
    }
};
METHOD
int main() {
    SdpaStableMaskCache cache;
    at::Tensor input({1,1,2,17},{at::kCPU,at::kFloat});
    for(int i=0;i<34;++i)(*input.data)[i]=float(i);
    auto first=sdpa_prepare_mask(input,false,2,17,cache,0);
    assert(publications==1 && first.mask_type==4 && first.ddr_tensor.shape==std::vector<int64_t>({2,32}));
    for(int r=0;r<2;++r)for(int c=0;c<32;++c) {
        float v=(*first.ddr_tensor.data)[r*32+c];
        if(c<17)assert(v==float(r*17+c));else assert(std::isinf(v) && v<0);
    }
    auto other=sdpa_prepare_mask(input,false,2,17,cache,1);
    assert(other.ddr_tensor.data!=first.ddr_tensor.data);
    (*input.data)[0]=99;
    auto replay=sdpa_prepare_mask(input,false,2,17,cache,0);
    assert(replay.ddr_tensor.data==first.ddr_tensor.data && (*first.ddr_tensor.data)[0]==99);
    assert((*other.ddr_tensor.data)[0]==0 && publications==3);
    at::Tensor resident({2,16},{at::kPrivateUse1,at::kHalf});
    sdpa_prepare_mask(resident,false,2,16,cache,0);
    assert(downloads==1 && publications==4);
    assert(sdpa_prepare_mask(input,true,2,17,cache,0).mask_type==1);
    assert(sdpa_prepare_mask({},false,2,17,cache,0).mask_type==0);
    bool rejected=false;try{sdpa_prepare_mask(input,false,3,17,cache,0);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected && publications==4);
}
'''.replace("METHOD", method)
    path, binary = tmp_path / "mask.cpp", tmp_path / "mask"
    path.write_text(program)
    subprocess.run([compiler, "-std=c++17", str(path), "-o", str(binary)], check=True,
                   capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
