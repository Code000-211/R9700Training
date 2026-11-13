Serving patches: 02xx-<topic>.patch. What a radiance needs to serve an adapter trained
here (LoRA support) and to answer a prefill with it, so a serving-only build can be
checked. Every patch whose name sorts into this set is applied, across both sets, in
file-name order, to a copy of the submodule in the build tree. No patch here observes
an op or replays a block: those stay in the training set.

A patch is a unified diff (git apply, -p1) beginning with two header lines:

    Concern: <the one thing it changes>
    Upstream: not-submitted | open | merged
