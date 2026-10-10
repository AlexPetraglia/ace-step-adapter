import pytest

from ace_step_adapter.parsing import RequestError, parse_release_request


def base(**overrides):
    body = {"prompt": "bossa nova, soft guitar", "lyrics": "[Verse]\nLine one", "audio_duration": 30}
    body.update(overrides)
    return body


def test_defaults_match_the_contract():
    parsed = parse_release_request({"prompt": "pop", "lyrics": "la la"})
    assert parsed.duration_seconds == 120
    assert parsed.vocal_language == "en"
    assert parsed.audio_format == "mp3"
    assert parsed.seed == -1


def test_caption_is_accepted_as_prompt_alias():
    parsed = parse_release_request({"caption": "rock", "lyrics": "x"})
    assert parsed.prompt == "rock"


def test_duration_alias_is_accepted():
    assert parse_release_request(base(audio_duration=None, duration=45)).duration_seconds == 45


@pytest.mark.parametrize("duration", [9, 601])
def test_duration_outside_range_is_rejected(duration):
    with pytest.raises(RequestError, match="audio_duration"):
        parse_release_request(base(audio_duration=duration))


def test_duration_must_be_integer():
    with pytest.raises(RequestError, match="integer"):
        parse_release_request(base(audio_duration="30"))


def test_lyrics_are_required():
    with pytest.raises(RequestError, match="lyrics"):
        parse_release_request({"prompt": "pop", "lyrics": "   "})


def test_prompt_is_required():
    with pytest.raises(RequestError, match="prompt"):
        parse_release_request({"lyrics": "x"})


@pytest.mark.parametrize(
    "flag",
    ["thinking", "use_format", "format", "use_cot_caption", "use_cot_language", "sample_mode"],
)
def test_language_model_options_are_rejected(flag):
    with pytest.raises(RequestError, match="not supported"):
        parse_release_request(base(**{flag: True}))


def test_disabled_language_model_options_are_accepted():
    parse_release_request(base(thinking=False, use_format=False, sample_mode=False))


@pytest.mark.parametrize("batch", [2, 0, "1"])
def test_batch_size_must_be_one(batch):
    with pytest.raises(RequestError, match="batch_size"):
        parse_release_request(base(batch_size=batch))


def test_unknown_audio_format_is_rejected():
    with pytest.raises(RequestError, match="audio_format"):
        parse_release_request(base(audio_format="ogg-vorbis"))


def test_fixed_seed_is_used_when_random_seed_is_disabled():
    parsed = parse_release_request(base(use_random_seed=False, seed=7))
    assert parsed.seed == 7


def test_fixed_seed_requires_a_value():
    with pytest.raises(RequestError, match="seed"):
        parse_release_request(base(use_random_seed=False))


def test_body_must_be_an_object():
    with pytest.raises(RequestError, match="JSON object"):
        parse_release_request(["not", "a", "dict"])
