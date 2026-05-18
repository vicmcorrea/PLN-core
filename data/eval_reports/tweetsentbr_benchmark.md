# TweetSentBR symbolic benchmark

Balanced 3-class test set: up to 600 tweets per class (seed=42), 1,703 examples in total.

| analyzer | n | acc | F1m | F1 pos | F1 neg | F1 neu | time | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority baseline | 1703 | 0.352 | 0.174 | 0.521 | 0.000 | 0.000 | 0.0s | always predicts the majority class |
| seed lexicon | 1703 | 0.391 | 0.326 | 0.342 | 0.146 | 0.489 | 0.0s | 30-word didactic seed lexicon + regex tokenizer |
| OpLexicon (regex) | 1703 | 0.424 | 0.426 | 0.454 | 0.421 | 0.403 | 0.0s | OpLexicon v3.0 + regex tokenizer |
| OpLexicon (spaCy lemmas) | 1703 | 0.437 | 0.438 | 0.473 | 0.426 | 0.414 | 6.7s | OpLexicon v3.0 + spaCy pt_core_news_sm |
| LeIA (Almeida 2018, PT-BR VADER) | 1703 | 0.500 | 0.500 | 0.482 | 0.545 | 0.473 | 0.2s | external lexicon-based tool, default thresholds (±0.05) |
| OpLexicon + tweet rules | 1703 | 0.510 | 0.505 | 0.601 | 0.500 | 0.415 | 0.0s | OpLexicon v3.0 + slang/emoji + tweet normalization |
| OpLexicon + SentiLex + tweet (best) | 1703 | 0.524 | 0.512 | 0.606 | 0.542 | 0.388 | 0.0s | Multi-lexicon fusion (OpLexicon + SentiLex-PT 02 + slang/emoji) |
