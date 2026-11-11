# Chat JSONL to token ids

`include/r97/chatdata.h` and `src/chatdata.cpp`: one `{"messages": [...]}` line per
conversation in, one `TokenSeq` out with a mask over the tokens the model should
be trained to produce. The format is the one `docs/DATA.md` describes for the
UltraChat placeholder.

```json
{"messages": [{"role": "system", "content": "You are terse."},
              {"role": "user", "content": "Hi"},
              {"role": "assistant", "content": "Hello!"}]}
```

A message's `role` and `content` must be strings; anything else is an error
naming the line. Roles are passed to the template untouched, so a template that
knows other roles can use them.

## Rendering

`ChatData` carries no template and no tokenizer of its own. `open()` reads the
container's vocabulary section and loads a `rad::Tokenizer` and a
`rad::ChatTemplate` from it (`core/text/tokenizer.h`, `core/text/chat.h`), the
same code the radiance server renders a request with, so training data and
served prompts tokenise identically. The conversation is rendered with
`ChatOptions`' defaults; in particular `add_generation_prompt = false`, because
the whole conversation is being rendered, not a request for the next turn. The
BOS rule is the one the server uses: a BOS token is added only when the
container asks for one and the render does not already start with it.

`r97train data --model <file.rad> --data <file.jsonl> [--show N]` prints the
statistics below and then the first `N` kept conversations, with each scored
token wrapped in `[[...]]` so the mask is visible in the rendered text.

## The mask

A token is scored when it is one the model would have generated:

1. for every assistant message, the tokens between the generation prompt that
   ends at that turn's header and the end of the turn:
   - the header is `apply(messages[0..i), add_generation_prompt = true)`, what
     serving feeds the model for that turn,
   - the end is `apply(messages[0..i], add_generation_prompt = false)`, the
     render that includes the turn,
   - the turn's span is the part of the full render between the two, located by
     their common token prefix rather than by counting characters;
2. nothing else. `score` is 0 on the system text, on every user turn, and on the
   template's role markers and separators, and `score[0]` is always 0.

The scored span is trimmed back to its last control token, so ChatML's
`<|im_end|>` is a target but the newline the template writes after it is not:
the model stops at the end-of-turn token, the newline belongs to the next
turn's header.

With a template that enables thinking by default -- the Qwen3.5 container's
template is one -- the assistant header ends inside the think block, and the
tokens the model generates to close the empty block fall inside the scored span,
because serving with the same options would have the model generate them too.
Rendering a conversation whose answers carry no reasoning therefore teaches an
empty think block; a dataset of reasoning traces needs no change, and a
non-thinking target needs `ChatOptions::enable_thinking = false`, not a mask
rule.

## Length

`kDefaultMaxTokens` is 4096. A conversation longer than the limit is shortened
by dropping trailing turns whole, starting from the last assistant turn, until
it fits; the result is the longest prefix that ends in an assistant turn. Two
cases produce nothing and are counted as skipped: a conversation with no
assistant turn (there is nothing to score, and trailing messages after the last
assistant turn are left out anyway), and one whose first exchange alone is too
long. `ChatDataStats` reports `conversations` (JSONL lines that held a conversation),
`kept` (including shortened ones), `shortened`, `skipped`, and the totals over
the kept sequences: `tokens` and `scored` (tokens with `score == 1`).

## Checked by

`tests/host/chatdata_mask.cpp` builds the expected ids and mask from the
fixture's ChatML template as text and its byte-level tokenizer, not from
`ChatData`: one-turn, three-turn and system-message conversations must match
exactly, a conversation limited to exactly two turns must drop the third whole,
one token less must keep only the first exchange, one token less than that must
skip, and `tests/fixtures/chat_small.jsonl` must read back the same three
sequences with the same totals. `tests/model/chatdata_render.cpp` renders the
same messages on `qwen3.8-27b-iq4r-s-v4.rad` and requires the ids to equal
radiance's own `ChatTemplate::apply` output for that request, the mask to be
non-empty, start unscored, and end before the render's trailing separator.

Both tests run inside the binary's `main` (which calls `r97test::run_all()`),
not from an exit handler: loading a chat template compiles the vendored chat
parser, whose namespace-scope tables are destroyed before any `atexit` handler
runs, so a render at exit fails with a `std::bad_array_new_length` from inside
radiance.
