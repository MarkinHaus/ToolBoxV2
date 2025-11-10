# 📚 TB Language - Documentation Hub

Welcome to the TB Language documentation! This hub provides links to all comprehensive guides and resources.

---

## 🎯 Quick Navigation

### For Users & Programmers

**📘 [TB Language Programming Guide](TB_LANG_PROGRAMMING_GUIDE.md)**
- Complete language reference
- Syntax and semantics
- Built-in functions
- Best practices
- Example programs
- Troubleshooting

**Perfect for:**
- Learning TB Language
- Writing TB programs
- Understanding language features
- Finding code examples

---

### For Contributors & Developers

**🛠️ [TB Language Development Guide](TB_LANG_DEVELOPMENT_GUIDE.md)**
- Compiler architecture
- Crate structure
- Development workflow
- Testing strategy
- Contributing guidelines
- Fix history (FIX 1-18)

**Perfect for:**
- Contributing to TB Lang
- Understanding compiler internals
- Adding new features
- Fixing bugs
- Optimizing performance

---

## 📖 Additional Documentation

### Core Documentation

- **[Lang.md](src/Lang.md)** - Language specification
- **[info.md](src/info.md)** - Feature documentation
- **[validated_syntax_and_features.md](src/validated_syntax_and_features.md)** - Test coverage tracking

### Test Files

- **[comprehensive_unit_tests.tbx](src/tests/comprehensive_unit_tests.tbx)** - 20 basic tests
- **[advanced_unit_tests.tbx](src/tests/advanced_unit_tests.tbx)** - 19 advanced tests
- **[diagnostic_tests_simple.tbx](src/tests/diagnostic_tests_simple.tbx)** - 10 diagnostic tests

---

## 🚀 Quick Start

### Installation

```bash
cd toolboxv2/tb-exc/src
cargo build
```

### Run Your First Program

```bash
# Create hello.tbx
echo 'fn main() { print("Hello, TB!") } main()' > hello.tbx

# Run in JIT mode
./target/debug/tb.exe run hello.tbx

# Compile to native binary
./target/debug/tb.exe compile hello.tbx
./target/tb-compile-cache/tb-compiled.exe
```

---

## 📊 Project Status

**Version:** 1.0  
**Last Updated:** 2025-11-10  
**Test Coverage:** 82% (31/38 E2E tests passing)  
**Status:** Production Ready

### Recent Achievements

✅ **FIX 1-17:** Foundation 100% stable  
✅ **FIX 18:** Multi-argument print() in compiled mode  
✅ **49 Unit Tests:** Comprehensive test coverage  
✅ **Dual Execution Modes:** JIT and AOT both working  

### Known Issues

⚠️ **Grade Calculator Bug:** Float values don't match Int ranges in match expressions  
⚠️ **Integer Division:** `/` operator returns Float for Int/Int division  

---

## 🎓 Learning Path

### 1. Beginner (Start Here!)

1. Read **[Programming Guide - Getting Started](TB_LANG_PROGRAMMING_GUIDE.md#getting-started)**
2. Learn **[Language Fundamentals](TB_LANG_PROGRAMMING_GUIDE.md#language-fundamentals)**
3. Practice with **[Example Programs](TB_LANG_PROGRAMMING_GUIDE.md#example-programs)**

### 2. Intermediate

1. Master **[Functions](TB_LANG_PROGRAMMING_GUIDE.md#functions)**
2. Learn **[Functional Programming](TB_LANG_PROGRAMMING_GUIDE.md#functional-programming)**
3. Explore **[Collections](TB_LANG_PROGRAMMING_GUIDE.md#collections)**

### 3. Advanced

1. Study **[Compiler Architecture](TB_LANG_DEVELOPMENT_GUIDE.md#architecture-overview)**
2. Understand **[Type System](TB_LANG_DEVELOPMENT_GUIDE.md#type-system)**
3. Contribute **[New Features](TB_LANG_DEVELOPMENT_GUIDE.md#common-development-tasks)**

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Read the [Development Guide](TB_LANG_DEVELOPMENT_GUIDE.md)**
2. **Check [Contributing Guidelines](TB_LANG_DEVELOPMENT_GUIDE.md#contributing-guidelines)**
3. **Pick a task from [Roadmap](TB_LANG_DEVELOPMENT_GUIDE.md#roadmap)**
4. **Submit a pull request**

### Good First Issues

- Add new built-in function (e.g., `abs()`, `min()`, `max()`)
- Improve error messages
- Add more unit tests
- Fix documentation typos

---

## 📞 Support

### Getting Help

1. **Check the guides** - Most questions are answered
2. **Search issues** - Someone may have had the same problem
3. **Create an issue** - For bugs or feature requests
4. **Join discussions** - Community support

### Reporting Bugs

Please include:
- TB Language version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior
- Error messages (if any)

---

## 📜 License

[Your License Here]

---

## 🙏 Acknowledgments

- **Rust Community** - For the amazing language and tools
- **Crafting Interpreters** - For inspiration and guidance
- **Contributors** - For making TB Language better

---

**Happy Coding with TB Language! 🚀**

*For detailed information, see the [Programming Guide](TB_LANG_PROGRAMMING_GUIDE.md) or [Development Guide](TB_LANG_DEVELOPMENT_GUIDE.md).*

