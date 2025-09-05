# Contributing to HAG-MoE

We're thrilled that you're interested in contributing to the Hierarchical Attention-Gated Mixture of Experts project! This document provides guidelines and information for contributors.

## 🌟 Ways to Contribute

- **🐛 Bug Reports**: Help us identify and fix issues
- **💡 Feature Requests**: Suggest new capabilities and improvements  
- **🔬 Research Contributions**: Share experimental results and theoretical insights
- **📖 Documentation**: Improve guides, examples, and API documentation
- **💻 Code Contributions**: Implement features, optimizations, and fixes
- **🧪 Testing**: Add test cases and improve test coverage

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/hag-moe.git
   cd hag-moe
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv hagmoe-dev
   source hagmoe-dev/bin/activate  # On Windows: hagmoe-dev\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

### Pre-commit Setup

We use pre-commit hooks to maintain code quality:

```bash
pre-commit install
```

This will automatically run formatting and linting checks before each commit.

## 📝 Contribution Guidelines

### Code Style

- **Python**: Follow PEP 8 with line length of 88 characters
- **Formatting**: Use `black` for code formatting
- **Imports**: Use `isort` for import sorting
- **Type Hints**: Include type hints for all function parameters and returns
- **Docstrings**: Use Google-style docstrings

Example:
```python
def expert_forward(
    self, 
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Forward pass through expert layer.
    
    Args:
        hidden_states: Input tensor of shape (batch_size, seq_len, d_model)
        attention_mask: Optional attention mask
        
    Returns:
        Tuple of (output_tensor, expert_metrics)
    """
    # Implementation here
    pass
```

### Commit Messages

Use conventional commit format:
- `feat:` new features
- `fix:` bug fixes  
- `docs:` documentation changes
- `test:` adding tests
- `refactor:` code refactoring
- `perf:` performance improvements

Example: `feat: add hierarchical expert routing mechanism`

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   pytest tests/
   black src/
   isort src/
   mypy src/
   ```

4. **Submit Pull Request**
   - Provide clear description of changes
   - Reference related issues
   - Include test results and benchmarks if applicable

## 🧪 Testing Guidelines

### Test Structure
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark critical operations

### Writing Tests
```python
import pytest
import torch
from hag_moe.models import HAGMoELayer

class TestHAGMoELayer:
    def test_forward_pass(self):
        """Test basic forward pass functionality."""
        layer = HAGMoELayer(d_model=512, num_experts=8)
        input_tensor = torch.randn(2, 10, 512)
        
        output, metrics = layer(input_tensor)
        
        assert output.shape == input_tensor.shape
        assert 'expert_utilization' in metrics
        assert metrics['expert_utilization'].sum() > 0
```

### Running Specific Tests
```bash
# Run specific test file
pytest tests/test_models/test_hag_moe.py -v

# Run with coverage
pytest tests/ --cov=hag_moe --cov-report=html
```

## 📚 Documentation

### API Documentation
- Use clear, descriptive docstrings
- Include parameter types and descriptions
- Provide usage examples
- Document expected behaviors and edge cases

### Examples and Tutorials
- Create practical, runnable examples
- Explain the reasoning behind design choices
- Include performance considerations
- Show integration with popular frameworks

## 🔬 Research Contributions

### Experimental Results
When contributing experimental results:
- Include reproducible code and configurations
- Provide detailed methodology descriptions
- Share raw data and analysis scripts
- Compare against relevant baselines

### Theoretical Contributions
- Provide mathematical proofs and derivations
- Include complexity analyses
- Reference related work appropriately
- Explain practical implications

## 🐛 Bug Reports

Please include:
- **Environment**: Python version, PyTorch version, OS
- **Reproduction Steps**: Minimal code to reproduce the issue
- **Expected vs Actual Behavior**: Clear description of the problem
- **Error Messages**: Full stack traces when applicable

Use this template:
```markdown
## Bug Description
Brief description of the issue

## Environment
- Python: 3.9.0
- PyTorch: 2.0.1
- HAG-MoE: 0.1.0
- OS: Ubuntu 22.04

## Reproduction Code
```python
# Minimal reproducible example
```

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Additional Context
Any other relevant information
```

## 💡 Feature Requests

For feature requests, please:
- Describe the motivation and use case
- Provide implementation suggestions if possible
- Consider backward compatibility
- Discuss potential performance implications

## 🏆 Recognition

Contributors will be recognized in:
- README acknowledgments
- Release notes
- Academic papers (for significant research contributions)
- Contributor hall of fame

## 📞 Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Create issues for bugs and feature requests
- **Email**: Contact maintainers for sensitive topics
- **Community**: Join our Discord/Slack (links in README)

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and constructive
- Focus on the technical aspects
- Help others learn and grow
- Report any unacceptable behavior

## 🎯 Development Priorities

Current focus areas:
1. **Performance Optimization**: Improving training and inference speed
2. **Memory Efficiency**: Reducing memory footprint
3. **Multimodal Support**: Extending to vision and other modalities
4. **Theoretical Analysis**: Deeper mathematical understanding
5. **Practical Applications**: Real-world use cases and benchmarks

Thank you for contributing to HAG-MoE! Together, we're pushing the boundaries of AI through elegant expert orchestration. 🚀
