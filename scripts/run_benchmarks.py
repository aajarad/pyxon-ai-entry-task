"""Run benchmark suite."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import db_manager
from src.benchmarks.suite import BenchmarkSuite


def main():
    """Run all benchmarks."""
    print("Running benchmark suite...")
    print("=" * 50)
    
    # Initialize database
    db_manager.init_db()
    session = db_manager.get_session()
    
    # Create benchmark suite
    suite = BenchmarkSuite(session)
    
    # Run all benchmarks
    results = suite.run_all_benchmarks()
    
    # Print results
    print("\nBENCHMARK RESULTS")
    print("=" * 50)
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    avg_score = sum(r.score for r in results) / total
    
    print(f"\nSummary: {passed}/{total} tests passed")
    print(f"Average Score: {avg_score:.2f}%")
    print()
    
    for result in results:
        status = "✓ PASSED" if result.passed else "✗ FAILED"
        print(f"{result.test_name}: {status}")
        print(f"  Score: {result.score:.2f}%")
        print(f"  Time: {result.execution_time:.2f}s")
        print(f"  Details: {result.details}")
        print()
    
    # Generate report
    report = suite.generate_report()
    print("\nFULL REPORT")
    print("=" * 50)
    print(report)
    
    # Save report to file
    with open("benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\nReport saved to benchmark_report.md")
    
    session.close()


if __name__ == "__main__":
    main()
