# Knowledge schema

The knowledge directory is a compact navigation and synthesis layer over
public RhinoForge source and documentation. It does not replace those sources.

## File types

- `concepts/<slug>.md`: explain one reusable execution contract.
- `synthesis/<slug>.md`: combine several concepts into a task-oriented guide.
- `queries/<slug>.md`: preserve a reusable, source-linked question and answer.
- `raw/<slug>.md`: immutable, reviewed snapshots of public external sources.
- `INDEX.md`: list every maintained knowledge page and its intended use.
- `RETRIEVAL.md`: define the Markdown-first retrieval path and optional indexing
  boundary.
- `WORKFLOWS.md`: define how to query and update the knowledge base.
- `log.md`: record public knowledge-base maintenance only.

## Page contract

Every concept or synthesis page must contain:

1. a descriptive H1 title;
2. the problem or decision the page addresses;
3. concise invariants or a workflow;
4. failure symptoms or stop conditions when applicable; and
5. relative links to the public documents or source files supporting the
   claims.

Keep one fact in one page and link to it elsewhere. Prefer stable public
contracts over implementation trivia. When knowledge conflicts with public
source or API documentation, correct the knowledge page rather than treating
it as authority.

Do not import private notes, development timelines, restricted device-program
details, opaque asset-format details, credentials, machine paths, or diagnostic
payloads. Add a page only when it helps repeat a real development or debugging
task.

Query pages follow the same source-link and sanitization rules. Raw snapshots
must additionally record provenance and redistribution terms; they are inputs,
not authoritative RhinoForge conclusions.
