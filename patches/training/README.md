Training patches: 01xx-<topic>.patch. Runtime hooks only the trainer needs. Every patch
whose name sorts into this set is applied, across both sets, in file-name order, to a
copy of the submodule in the build tree (-DR97_PATCH_SET=all, the default). Serving-only
builds (-DR97_PATCH_SET=serving) apply none of these.

A patch is a unified diff (git apply, -p1) beginning with two header lines:

    Concern: <the one thing it changes>
    Upstream: not-submitted | open | merged
