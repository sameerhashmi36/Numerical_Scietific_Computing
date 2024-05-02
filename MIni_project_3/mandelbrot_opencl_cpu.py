import numpy as np
import pyopencl as cl
import time

def mandelbrot_opencl(width, height, xmin, xmax, ymin, ymax, max_iterations):
    try:
        # Load the OpenCL kernel code from a file
        with open("mandelbrot_opencl.cl", "r") as f:
            kernel_code = f.read()

        # Get all available platforms
        platforms = cl.get_platforms()
        print("Available platforms:", platforms)

        # Try to select GPU devices first, fall back to CPU if no GPUs are available
        devices = []
        for platform in platforms:
            try:
                devices.extend(platform.get_devices(device_type=cl.device_type.GPU))
            except cl.LogicError:
                continue  # Skip if no GPUs are found on this platform

        # If no GPUs were found, try to find CPUs
        if not devices:
            for platform in platforms:
                try:
                    devices.extend(platform.get_devices(device_type=cl.device_type.CPU))
                except cl.LogicError:
                    continue  # Skip if no CPUs are found on this platform
        
        if not devices:
            raise RuntimeError("No OpenCL-compatible GPU or CPU found.")

        print("Using devices:", devices)

        results = []

        for device in devices:
            try:
                # Create a context and a command queue for the device
                context = cl.Context([device])
                queue = cl.CommandQueue(context)

                # Create and build the OpenCL program
                program = cl.Program(context, kernel_code).build()

                # Allocate memory for the results on the device
                output_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, width * height * np.dtype(np.int32).itemsize)

                # Execute the kernel
                start_time = time.time()
                program.calculate_mandelbrot(queue, (width, height), None, output_buf,
                                             np.int32(width), np.int32(height), np.float32(xmin), np.float32(xmax),
                                             np.float32(ymin), np.float32(ymax), np.int32(max_iterations))
                queue.finish()  # Ensure all operations are completed
                end_time = time.time()

                # Retrieve the results from the device
                output = np.empty((height, width), dtype=np.int32)
                cl.enqueue_copy(queue, output, output_buf)

                execution_time = end_time - start_time
                print(f"Execution Time for {device.name}: {execution_time} seconds")
                results.append((device.name, execution_time))

            except cl.Error as e:
                print(f"OpenCL error processing device {device.name}: {e}")
            except Exception as e:
                print(f"General error processing device {device.name}: {e}")

        return results

    except Exception as e:
        print(f"An error occurred during setup or execution: {e}")
        return []

# Example usage: Benchmarking different grid sizes
widths = [500, 1000, 1500]
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5
max_iterations = 100

for width in widths:
    print(f"Benchmarking results for width={width}:")
    results = mandelbrot_opencl(width, width, x_min, x_max, y_min, y_max, max_iterations)
    for device_name, execution_time in results:
        print(f"Device: {device_name}, Execution Time: {execution_time} seconds")