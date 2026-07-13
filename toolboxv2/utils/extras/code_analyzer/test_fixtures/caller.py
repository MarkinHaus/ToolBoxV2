from toolboxv2.utils.extras.code_analyzer.test_fixtures.helpers import compute, transform, aggregate

def run_pipeline():
    results = []
    for i in range(20):
        v = compute(5000)
        results.append(v)
    final = transform(results)
    total = aggregate(final)
    return total

def run_nested():
    return aggregate(transform([compute(1000) for _ in range(10)]))
