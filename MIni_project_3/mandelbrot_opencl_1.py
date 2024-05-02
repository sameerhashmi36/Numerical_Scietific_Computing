import numpy as np
import pyopencl as cl
import time

def mandelbrot_opencl(width, height, xmin, xmax, ymin, ymax, max_iterations, device_type="GPU"):
    try:
        # Read the OpenCL kernel code
        with open("mandelbrot_opencl.cl", "r") as f:
            kernel_code = f.read()

        # Initialize OpenCL context and command queue
        platforms = cl.get_platforms()
        print("platforms:", platforms)
        devices = platforms[0].get_devices(cl.device_type.GPU)
        print("devices:", devices)

        if device_type == "CPU":
            devices = platforms[0].get_devices(cl.device_type.CPU)
        else:
            devices = platforms[0].get_devices(cl.device_type.GPU)

        print("devices2:", devices)

        results = []

        for device in devices:
            try:
                context = cl.Context([device])
                queue = cl.CommandQueue(context)

                # Create the OpenCL program
                program = cl.Program(context, kernel_code).build()

                # Allocate memory for output array on the device
                output_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, width * height * np.dtype(np.int32).itemsize)

                # Execute the kernel
                start_time = time.time()
                program.calculate_mandelbrot(queue, (width, height), None,
                                   output_buf, np.int32(width), np.int32(height), np.float32(xmin), np.float32(xmax),
                                   np.float32(ymin), np.float32(ymax), np.int32(max_iterations))
                queue.finish()
                end_time = time.time()

                # Read the result back from the device
                output = np.empty((height, width), dtype=np.int32)
                cl.enqueue_copy(queue, output, output_buf)

                execution_time = end_time - start_time
                print(f"Execution Time for {device.name}: {execution_time} seconds")
                results.append((device.name, execution_time))

            except Exception as e:
                print(f"Error processing device {device.name}: {e}")

        return results

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Benchmarking
widths = [500, 1000, 1500]  # Different grid sizes
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5
max_iterations = 100

for width in widths:
    print(f"Benchmarking results for width={width}:")
    results = mandelbrot_opencl(width, width, x_min, x_max, y_min, y_max, max_iterations)
    for device_name, execution_time in results:
        print(f"Device: {device_name}, Execution Time: {execution_time} seconds")
