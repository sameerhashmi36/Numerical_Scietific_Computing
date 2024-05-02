__kernel void calculate_mandelbrot(__global float *result, const int width, const int height, const float xmin, const float xmax, const float ymin, const float ymax, const int max_iterations) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    float x_coord = xmin + i * (xmax - xmin) / width;
    float y_coord = ymin + j * (ymax - ymin) / height;
    
    float x = 0.0f;
    float y = 0.0f;
    
    int iteration = 0;
    while (x*x + y*y < 4.0f && iteration < max_iterations) {
        float xtemp = x*x - y*y + x_coord;
        y = 2 * x*y + y_coord;
        x = xtemp;
        iteration++;
    }
    
    result[j * width + i] = iteration;
}
