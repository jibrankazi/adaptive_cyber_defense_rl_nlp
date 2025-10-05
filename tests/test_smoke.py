def test_imports():
    import importlib
    for m in ["src.train_rl", "src.train_nlp", "src.evaluate"]:
        importlib.import_module(m.replace("/", ".") if "/" in m else m)
