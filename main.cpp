#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include "fpng.h"

inline void blend_pixel(uint8_t* dest, const uint8_t intensity) {

    const double color = (static_cast<double>(*dest) / 255.f) * (static_cast<double>(intensity) / 255.f)*255.f;
        *dest = static_cast<uint8_t>(color);
        *(dest+1) = static_cast<uint8_t>(color);
        *(dest+2) = static_cast<uint8_t>(color);

}

void draw_aa_point(uint8_t* buffer, const int width, const int height, const float x, const float y,
                  const uint8_t color = 3, const float radius = 1.f) {
    // Calculate integer bounds
    const int x0 = std::max(0, static_cast<int>(std::floor(x - radius)));
    const int y0 = std::max(0, static_cast<int>(std::floor(y - radius)));
    const int x1 = std::min(width - 1, static_cast<int>(std::ceil(x + radius)));
    const int y1 = std::min(height - 1, static_cast<int>(std::ceil(y + radius)));

    const float r_squared = radius * radius;

    // Visit each affected pixel
    for (int py = y0; py <= y1; py++) {
        for (int px = x0; px <= x1; px++) {
            // Calculate squared distance from point to pixel center
            const float dx = static_cast<float>(px) - x;
            const float dy = static_cast<float>(py) - y;
            const float dist_squared = dx*dx + dy*dy;

            // Skip pixels outside the radius
            if (dist_squared > r_squared) {
                continue;
            }

            // Calculate alpha based on distance (smooth falloff)
            float alpha = 1.0f - std::sqrt(dist_squared) / radius;
            alpha = alpha * alpha; // Square for more natural falloff

            // Calculate blended intensity (higher alpha = closer to point color)
            const auto intensity = static_cast<uint8_t>(255 - color*alpha);

            // Blend with the existing pixel
            uint8_t* dest_pixel = &buffer[(py * width + px) * 3];
            blend_pixel(dest_pixel, intensity);
        }
    }
}

int main() {
    //Init the fpng library
    fpng::fpng_init();

    // Report the number of threads OpenMP will use
    const auto num_threads = omp_get_max_threads();
    std::cout << "Using " << num_threads << " threads" << std::endl;

    //PNG Path
    auto filename = "bifurcation.png";

    // Image settings
    constexpr size_t width  = 3840*4;      // PNG Width in pixels
    constexpr size_t height = 2160*4;      // PNG Height in pixels
    constexpr size_t r_steps   = width;   // Number of r values from start to end
    constexpr size_t max_iter  = 50000;  // Total number of iterations per r
    const size_t skip_iter = 4000;    // Iterations to skip (allow the orbit to settle)
    const float  aa_radius = 1.f;     // Antialiasing radius (smaller = sharper)
    const double log_scale_power = 0.25; // Power for logarithmic scaling (lower = more detail at high r values)
    constexpr double start = 1;         // start of r value
    constexpr double end = 4;           // end of r value

    const auto interval = end - start;

    // Use a vector of bytes for RGB image (initialized to white)
    auto *pixelBuffer = new uint8_t[width * height * 3];
    std::fill_n(pixelBuffer, (width * height * 3), 255);

    // Measure computation time
    double start_time = omp_get_wtime();
    const long long total_iters = r_steps*max_iter;

    // Parallelize the loop over r values with dynamic scheduling for better load balancing
    #pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < r_steps; ++i) {
        // Compute this thread's parameter value r with logarithmic scaling
        // This concentrates more points in the region where bifurcations become more complex
        const long double t = static_cast<long double>(i) / (r_steps - 1);

        // Power-based logarithmic scaling (adjustable concentration)
        const long double scaled_t = std::pow(t, log_scale_power);

        // Map to range [1.0, 4.0]
        const long double r = start + interval * scaled_t;

        // Thread-local point storage to avoid race conditions
        std::vector<std::pair<float, float>> points;

        // Start from some initial x
        long double x = 0.5;

        // Skip the first 'skip_iter' iterations (transient)
        for (int n = 0; n < skip_iter; ++n) {
            x = r * x * (1.0 - x);
        }

        // Now record the orbit points
        for (int n = skip_iter; n < max_iter; ++n) {
            x = r * x * (1.0 - x);
            // Convert (r, x) to image coordinates (using float for subpixel precision)
            // For x coordinate, we map from logarithmic r-space back to linear image space
            // This ensures uniform distribution in the output image
            float px = static_cast<float>(i);
            float py = static_cast<float>((1 - x) * (height/1.0f - 1));

            // Store the point if in range
            if (py >= 0 && py < height) {
                points.emplace_back(px, py);
            }
        }

        // Now draw all points for this r value with antialiasing
        for (const auto&[x, y] : points) {
                draw_aa_point(pixelBuffer, width, height, x, y, 1, aa_radius);

        }
    }

    const double compute_time = omp_get_wtime() - start_time;
    std::cout << "Computation time: " << compute_time << " seconds" << std::endl;
    std::cout << "Total iterations: " << total_iters << std::endl;
    std::cout << static_cast<int>(total_iters / compute_time) << " iterations/second" << std::endl << std::endl;

    // Measure rendering time
    start_time = omp_get_wtime();

    // Enhance contrast with a simple gamma correction
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t* pixel = &pixelBuffer[(y * width + x) * 3];
            // Apply gamma correction to enhance contrast
            float value = (255.0f - pixel[0]) / 255.0f;
            value = std::pow(value, 0.7f); // Gamma < 1 enhances dark details
            pixel[0] = pixel[1] = pixel[2] = (static_cast<uint8_t>(value * 200.0f));
        }
    }

    double render_time = omp_get_wtime() - start_time;
    std::cout << "Rendering time: " << render_time << " seconds" << std::endl;
    std::cout << "Total time: " << compute_time + render_time << " seconds" << std::endl;

    if (fpng::fpng_encode_image_to_file(filename, pixelBuffer, width, height, 3)) {
        std::cout << "Bifurcation diagram saved to " << filename << std::endl;
    }

    return 0;
}