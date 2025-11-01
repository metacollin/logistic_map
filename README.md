Demo C++ code that generates the logistic map. Uses a modified version of fpng that supports gigapixel PNG generation.

To build (requires a C++ compiler with `OpenMP` built-in like `clang` as well as `cmake`):

```bash
git clone https://github.com/metacollin/logistic_map.git
cd logistic_map
mkdir build
cd build
cmake ..
make
```

This will generate a png named `bifurcation.png`. Here is the example output (Note: image has been downscaled from original). To get something else, modify constexpr settings in the source code and recompile:
![logistic map][logistic_map]

[logistic_map]: https://github.com/metacollin/logistic_map/raw/main/bifurcation.png "Logo Title Text 2"
