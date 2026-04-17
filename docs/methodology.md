# methodology

this file explains the current baseline at a high level, what was created by us, and what comes from an external lexical resource.

for a more detailed step by step breakdown of the pipeline, see `docs/pipeline-processes.md`.

for the new per process documentation set, see `docs/processes/00_index.md`.

## 1. lexicon source

the project currently supports two lexical sources.

### 1.1 built-in seed dictionary

the built-in dictionary was created manually by us for the baseline. it was not mined from a corpus and it was not copied from a public sentiment lexicon.

the idea was simple:

- pick a small set of common brazilian portuguese words that clearly carry positive or negative sentiment
- assign a manual score by perceived intensity
- keep the scale small and interpretable, with weak words closer to zero and strong words farther from zero

examples:

- `bom`, `boa`, `legal` get small positive values
- `amei`, `otimo`, `maravilhoso` get stronger positive values
- `ruim`, `confuso`, `chato` get negative values
- `odiei`, `horrivel`, `pessimo` get stronger negative values

this is best described as a hand-built seed lexicon for a minimal symbolic prototype.

### 1.2 oplexicon v3.0

the second option uses `oplexicon v3.0`, which is an external portuguese sentiment lexicon published by the pucrs group.

this is different from our local lexicon in two important ways:

1. the local seed lexicon is manually written by us and intentionally small
2. `oplexicon` is an external resource with much broader lexical coverage and published polarity annotations

in the code, we load the file `lexico_v3.0.txt` format, whose rows follow this structure:

- `term`
- `type`
- `polarity`
- `polarity_revision`

for example, the project reads rows like:

- `bom,adj,1,A`
- `ruim,adj,-1,A`
- `:),emot,1,A`

the implementation downloads and caches the raw file locally the first time `oplexicon` is selected.

### 1.3 source selection flow

```mermaid
flowchart TD
    A[start cli] --> B{which lexicon?}
    B -->|seed| C[load local seed_lexicon.json]
    B -->|oplexicon| D[download or reuse cached oplexicon file]
    D --> E[parse lexico_v3.0.txt]
    C --> F[build analyzer]
    E --> F[build analyzer]
```

## 2. preprocessing and tokenization

before scoring, the text goes through lightweight normalization.

### 2.1 preprocessing

the current pipeline does this:

- remove urls
- remove mentions like `@user`
- keep hashtag words but drop the `#`
- reduce repeated characters of length 3 or more to length 2
- normalize extra whitespace

this means a noisy input such as:

`ameiiii esse filme!!! #perfeito`

becomes something closer to:

`ameii esse filme!!! perfeito`

### 2.2 tokenization choices

the project now supports two tokenizer backends.

#### custom tokenizer

the custom tokenizer is the main symbolic baseline. after normalization and accent folding, it extracts:

- alphanumeric word tokens
- basic emoticon tokens such as `:)`, `:(`, `=d`

this is deterministic and easy to explain in a symbolic project report.

#### spaCy portuguese tokenizer

the second option uses spaCy with `spacy.blank("pt")`.

this creates a tokenizer-only portuguese pipeline, so we get a real tokenizer without needing a large pretrained tagging or parsing model.

we still run the same project normalization first, then hand the normalized text to the spaCy tokenizer. this gives a more standard language-aware tokenizer option while keeping the rest of the symbolic pipeline unchanged.

this is different from our custom tokenizer because:

1. the custom tokenizer is fully handcrafted and regex based
2. the spaCy option uses portuguese-specific tokenization rules provided by spaCy
3. the custom tokenizer is easier to explain as a pure symbolic baseline
4. the spaCy tokenizer is useful as a more realistic off-the-shelf comparison

### 2.3 tokenization details

after normalization, the text is folded to lowercase and accents are removed for matching.

examples:

- `ótimo` becomes `otimo`
- `péssimo` becomes `pessimo`

with the custom tokenizer, the system extracts:

- alphanumeric word tokens
- basic emoticon tokens such as `:)`, `:(`, `=d`

with the nltk option, token boundaries are produced by `TweetTokenizer` after the same project normalization step.

### 2.4 tokenizer selection flow

```mermaid
flowchart TD
    A[normalized text] --> B{which tokenizer?}
    B -->|custom| C[regex tokenizer]
    B -->|spacy| D[spaCy blank pt tokenizer]
    C --> E[token list]
    D --> E[token list]
```

### 2.5 preprocessing flow

```mermaid
flowchart TD
    A[raw text] --> B[remove urls and mentions]
    B --> C[strip # but keep hashtag word]
    C --> D[compress repeated chars]
    D --> E[normalize whitespace]
    E --> F[lowercase and remove accents]
    F --> G[pass text to selected tokenizer]
```

## 3. lexical scoring

once we have tokens, we score only the tokens that appear in the chosen lexicon.

### 3.1 direct lookup

each token is searched in the selected lexicon:

- if found, start from its base score
- if not found, skip it

### 3.2 rule adjustments

after the base score is found, a few symbolic rules can change it.

the current rules are:

- negation: if a negation appears in the previous three tokens, flip the polarity
- intensifier: if the previous token is something like `muito` or `super`, increase the magnitude
- diminisher: if the previous token is something like `pouco` or `quase`, reduce the magnitude
- contrast: if a marker like `mas` appears, give more weight to the clause after it
- exclamation: `!` slightly increases emotional strength

### 3.3 scoring flow

```mermaid
flowchart TD
    A[token] --> B{token in lexicon?}
    B -->|no| C[skip token]
    B -->|yes| D[start with base polarity]
    D --> E{negation in recent context?}
    E -->|yes| F[flip sign]
    E -->|no| G[keep sign]
    F --> H[apply intensifier or diminisher]
    G --> H
    H --> I[adjust for contrast marker]
    I --> J[adjust for exclamation]
    J --> K[store adjusted token score]
```

## 4. final label

the final sentence score is the sum of all adjusted token scores.

then we assign a label:

- `positive` if score >= `0.75`
- `negative` if score <= `-0.75`
- `neutral` otherwise

if no lexical match is found at all, the result is also `neutral`.

### 4.1 decision flow

```mermaid
flowchart TD
    A[sum adjusted token scores] --> B{any matched term?}
    B -->|no| C[neutral]
    B -->|yes| D{score >= 0.75?}
    D -->|yes| E[positive]
    D -->|no| F{score <= -0.75?}
    F -->|yes| G[negative]
    F -->|no| H[neutral]
```

## 5. how to explain it in the report

if you need to explain the current baseline honestly, the cleanest wording is:

1. the project implements a symbolic sentiment analyzer for short portuguese texts
2. the baseline uses a manually created seed dictionary designed for prototyping and debugging
3. the system also supports an external lexical resource, `oplexicon v3.0`, for a broader symbolic lexicon
4. tokenization is rule based and regex based after lightweight text normalization
5. the final prediction comes from lexical polarity plus symbolic rules for negation, intensification, contrast, and punctuation

## 6. comparison mode

the cli also has a comparison mode for quick qualitative inspection.

instead of analyzing only one configuration, it runs the same example texts through four combinations:

- seed dictionary + custom tokenizer
- seed dictionary + spaCy portuguese tokenizer
- oplexicon + custom tokenizer
- oplexicon + spaCy portuguese tokenizer

this is useful for checking how lexical coverage and token boundaries affect the final label and score.

```mermaid
flowchart TD
    A[start comparison mode] --> B[load seed dictionary]
    A --> C[load oplexicon]
    B --> D[run custom tokenizer]
    B --> E[run spaCy tokenizer]
    C --> F[run custom tokenizer]
    C --> G[run spaCy tokenizer]
    D --> H[show side by side outputs]
    E --> H
    F --> H
    G --> H
```

## 7. important limitation

the built-in seed dictionary is intentionally small. it is useful for a minimal working prototype, but it is not enough to claim broad lexical coverage.

for a stronger experiment or evaluation, `oplexicon` should be the preferred lexical source.
