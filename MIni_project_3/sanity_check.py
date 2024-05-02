import numpy as np
import pyopencl as cl
import time

def print_platforms_and_devices():
    platforms = cl.get_platforms()
    print("Number of platforms:", len(platforms))
    for platform in platforms:
        print("Platform Name:", platform.name)
        print("Platform Vendor:", platform.vendor)
        print("Platform Version:", platform.version)
        print("Platform Profile:", platform.profile)
        print("Platform Extensions:", platform.extensions)
        print("")

        devices = platform.get_devices()
        print("Number of devices:", len(devices))
        for device in devices:
            print("  Device Name:", device.name)
            print("  Device Vendor:", device.vendor)
            print("  Device Vendor ID:", hex(device.vendor_id))
            print("  Device Version:", device.version)
            print("  Driver Version:", device.driver_version)
            print("  Device Type:", cl.device_type.to_string(device.type))
            print("  Device Profile:", device.profile)
            print("  Device Available:", "Yes" if device.available else "No")
            print("  Compiler Available:", "Yes" if device.compiler_available else "No")
            print("  Linker Available:", "Yes" if device.linker_available else "No")
            print("  Max compute units:", device.max_compute_units)
            print("  Max clock frequency:", device.max_clock_frequency, "MHz")
            print("  Max work item sizes:", device.max_work_item_sizes)
            print("  Max work group size:", device.max_work_group_size)
            print("  Preferred work group size multiple:", device.preferred_work_group_size_multiple)
            print("  Global memory size:", device.global_mem_size, "(Bytes)")
            print("  Max memory allocation size:", device.max_mem_alloc_size, "(Bytes)")
            print("  Local memory size:", device.local_mem_size, "(Bytes)")
            print("  Max constant buffer size:", device.max_constant_buffer_size, "(Bytes)")
            print("")

def mandelbrot_opencl(width, height, xmin, xmax, ymin, ymax, max_iterations, device_type="GPU"):
    # Read the OpenCL kernel code
    with open("mandelbrot_opencl.cl", "r") as f:
        kernel_code = f.read()

    # Get platforms and devices
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(cl.device_type.CPU if device_type == "CPU" else cl.device_type.GPU)

    # Print platforms and devices for debugging
    print("Number of platforms:", len(platforms))
    for platform in platforms:
        print("Platform Name:", platform.name)
    print("Number of devices:", len(devices))
    for device in devices:
        print("  Device Name:", device.name)

    # Explicitly specify platform and device
    platform_index = 0
    device_index = 0
    try:
        context = cl.Context(devices=[devices[device_index]], properties=[(cl.context_properties.PLATFORM, platforms[platform_index])])
    except cl.LogicError as e:
        print("Error creating context:", e)
        return None, None

    queue = cl.CommandQueue(context)

    # Create the OpenCL program
    try:
        program = cl.Program(context, kernel_code).build()
    except cl.BuildProgramFailure as e:
        print("Error building program:", e)
        return None, None

    # Allocate memory for output array on the device
    output_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, width * height * np.dtype(np.int32).itemsize)

    # Execute the kernel
    start_time = time.time()
    try:
        program.calculate_mandelbrot(queue, (width, height), None,
                                     output_buf, np.int32(width), np.int32(height), np.float32(xmin), np.float32(xmax),
                                     np.float32(ymin), np.float32(ymax), np.int32(max_iterations))
    except cl.LogicError as e:
        print("Error executing kernel:", e)
        return None, None

    queue.finish()
    end_time = time.time()

    # Read the result back from the device
    output = np.empty((height, width), dtype=np.int32)
    cl.enqueue_copy(queue, output, output_buf)

    # Return the Mandelbrot set and execution time
    return output, end_time - start_time

width, height = 1000, 1000
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5
max_iterations = 100

output, execution_time = mandelbrot_opencl(width, height, x_min, x_max, y_min, y_max, max_iterations)
if output is not None:
    print("Execution time:", execution_time, "seconds")