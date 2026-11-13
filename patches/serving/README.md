Serving patches: 02xx-<topic>.patch. What a radiance needs to serve an adapter trained
here (LoRA support). Every patch whose name sorts into this set is applied, across both
sets, in file-name order, to a copy of the submodule in the build tree.

A patch is a unified diff (git apply, -p1) beginning with two header lines:

    Concern: <the one thing it changes>
    Upstream: not-submitted | open | merged
