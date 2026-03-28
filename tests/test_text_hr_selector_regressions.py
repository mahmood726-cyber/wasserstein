import robust_km_pipeline as rkp


class _DummyPage:
    def __init__(self, text):
        self._text = text

    def get_text(self):
        return self._text


class _DummyDoc:
    def __init__(self, text):
        self._pages = [_DummyPage(text)]

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, idx):
        return self._pages[idx]

    def close(self):
        return None


def _extract_text_hr(monkeypatch, text):
    monkeypatch.setattr(rkp.fitz, "open", lambda _path: _DummyDoc(text))
    pipeline = rkp.RobustKMPipeline()
    hr, ctx = pipeline._extract_text_hr("dummy.pdf")
    return hr, ctx


def test_text_selector_prefers_not_not_statistically_different_candidate(monkeypatch):
    text = (
        "The treatment effect was not statistically different for TNFi versus JAKi "
        "(csHR 1.00, 95% CI 0.92 to 1.10). "
        "The hazard was higher for OMA versus JAKi (csHR 1.11, 95% CI 1.01 to 1.23)."
    )
    hr, _ = _extract_text_hr(monkeypatch, text)
    assert hr == 1.11


def test_text_selector_penalizes_high_p_value_when_better_candidate_exists(monkeypatch):
    text = (
        "All-cause mortality showed HR 1.17, 95% CI 0.51-2.71, P = 0.71. "
        "Composite endpoint showed HR 0.51, 95% CI 0.29-0.86, P = 0.012."
    )
    hr, _ = _extract_text_hr(monkeypatch, text)
    assert hr == 0.51


def test_text_selector_treats_hr_95ci_pattern_as_ci_candidate(monkeypatch):
    text = (
        "Abstract reports hazard ratio 0.90. "
        "Primary endpoint HR (95% CI) 1.20 (1.10-1.30)."
    )
    hr, _ = _extract_text_hr(monkeypatch, text)
    assert hr == 1.2
