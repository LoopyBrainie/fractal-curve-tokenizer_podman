# Fractal Curve Tokenizer Tests

This directory contains comprehensive tests for the fractal curve tokenizer and fractal ViT implementation.

## Test Structure

### Unit Tests (`unit_tests/`)
- `test_fractal_hilbert_tokenizer.py`: Core tokenizer functionality tests
- `test_hilbert_effectiveness.py`: Hilbert curve effectiveness evaluation
- `test_hilbert_verification.py`: Hilbert curve correctness verification
- `test_improved_hilbert.py`: Enhanced Hilbert implementation tests

### Integration Tests (`integration_tests/`)
- `test_enhanced_fractal_vit.py`: Full model integration testing
- `test_fractal_zigzag_manim.py`: Visualization and animation tests

### Benchmarks (`benchmarks/`)
- `benchmark_fractal_vit.py`: Performance benchmarking
- `compare_fractal_vs_standard.py`: Comparative analysis with standard ViT

### Training Scripts (`training_scripts/`)
- `train_fractal_vit_complete.py`: Complete fractal ViT training pipeline
- `train_enhanced_fractal_vit.py`: Enhanced model training
- `train_fractal_vit.py`: Basic fractal ViT training
- `train_fractal_cifar10.py`: CIFAR-10 specific training

## Running Tests

### Unit Tests
```bash
python -m pytest tests/unit_tests/ -v
```

### Integration Tests
```bash
python -m pytest tests/integration_tests/ -v
```

### Run Specific Test
```bash
python tests/unit_tests/test_fractal_hilbert_tokenizer.py
```

### Benchmarks
```bash
python tests/benchmarks/benchmark_fractal_vit.py
python tests/benchmarks/compare_fractal_vs_standard.py
```

### Training Scripts
```bash
python tests/training_scripts/train_fractal_vit_complete.py
```

## Test Categories

1. **Functionality Tests**: Verify core tokenizer operations
2. **Performance Tests**: Measure speed and efficiency
3. **Correctness Tests**: Validate Hilbert curve properties
4. **Integration Tests**: Test full model pipeline
5. **Comparative Tests**: Compare with standard implementations
