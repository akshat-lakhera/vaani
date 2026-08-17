from vaani.guardrails import generated_is_grounded, grounding, input_guard, off_topic


def test_refuses_password():
    d = input_guard("What is my bank account password?")
    assert not d.ok and d.status == "refuse"


def test_refuses_hindi_password():
    d = input_guard("मेरे बैंक खाते का पासवर्ड क्या है?")
    assert not d.ok and d.status == "refuse"


def test_allows_capital_question():
    d = input_guard("भारत की राजधानी क्या है?")
    assert d.ok


def test_empty_refused():
    assert input_guard("   ").status == "refuse"


def test_off_topic_threshold():
    assert off_topic(0.05, 0.22).status == "abstain"
    assert off_topic(0.55, 0.22).ok


def test_grounding_requires_substring():
    ctx = ["दिल्ली भारत की राजधानी है।"]
    ok = grounding("दिल्ली भारत की राजधानी है।", ctx, 0.4)
    assert ok.ok
    bad = grounding("पेरिस फ्रांस की राजधानी है।", ctx, 0.4)
    assert not bad.ok


def test_coverage_gate_catches_term_collision():
    from vaani.guardrails import coverage_gate

    weather_ny = ["न्यूयॉर्क में मार्च में मौसम कैसा है? यह पोस्ट न्यूयॉर्क शहर के मार्च मौसम का सारांश है।"]
    assert not coverage_gate("आज गोवा में मौसम कैसा है?", weather_ny, 0.6).ok
    corp = ["एक निगम एक कंपनी या लोगों का समूह है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत है।"]
    # "कॉर्पोरेशन" vs "निगम" — different words; coverage should fail
    assert not coverage_gate("कॉर्पोरेशन क्या है?", corp, 0.6).ok


def test_generated_verify():
    ctx = ["The capital of India is New Delhi."]
    assert generated_is_grounded("The capital of India is New Delhi.", ctx)
    assert not generated_is_grounded("The capital of France is Paris and also Berlin.", ctx)
