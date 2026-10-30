import re

_WORD_RE = re.compile(r"[а-яА-ЯёЁa-zA-Z0-9]+")

_STOPWORDS = frozenset(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от "
    "меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж "
    "вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без "
    "будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один "
    "почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после "
    "над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свой этой перед "
    "иногда лучше чуть том нельзя такой им более всегда конечно всю между".split()
)

_morph = None
_lemma_cache: dict[str, str] = {}


def _analyzer():
    global _morph
    if _morph is None:
        import pymorphy3

        _morph = pymorphy3.MorphAnalyzer()
    return _morph


def tokenize_lemma(text: str) -> list[str]:
    morph = _analyzer()
    out = []
    for tok in _WORD_RE.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        if not tok.isalpha() or not re.search(r"[а-яё]", tok):
            out.append(tok)
            continue
        lemma = _lemma_cache.get(tok)
        if lemma is None:
            lemma = morph.parse(tok)[0].normal_form
            _lemma_cache[tok] = lemma
        if lemma not in _STOPWORDS:
            out.append(lemma)
    return out
