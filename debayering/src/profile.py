def profiler(start_time, end_time, source_np):
    h, w, _ = source_np.shape
    time = end_time - start_time
    
    return time, time/h/w * 10**6
