# Knowledge retrieval

The checked-in Markdown files are the source of truth. No embedding service or
external index is required to use this knowledge base.

## Default retrieval path

1. Start at [INDEX.md](INDEX.md).
2. Read the smallest matching concept or synthesis page.
3. Follow its relative links to public documentation or source before making a
   support claim or changing code.
4. Use `rg` across public source only when the indexed pages do not answer the
   question.

Save a reusable, source-linked answer under [queries/](queries/README.md) only
when it is likely to be asked again. Move any general conclusion into the
owning concept or synthesis page so future retrieval does not depend on the
original question wording.

## Optional semantic indexing

Add a semantic index only when navigation no longer provides adequate recall,
or when several tools need a shared query endpoint. The index is derived and
rebuildable: Markdown remains authoritative and every answer still needs links
back to public sources.

Before sending content to an embedding or hosted retrieval provider, confirm
that its data-handling terms permit the repository content. Exclude credentials,
private infrastructure, restricted implementation material, diagnostic
artifacts, and unpublished model inputs. Store approved external source
snapshots under [raw/](raw/README.md); do not copy repository files there.
