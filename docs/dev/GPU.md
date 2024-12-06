---
time: 2024-07-11 13:58
tags:
  - pytorch
---
# GPU

## Overview



## Details

### Memory:
#### Overview:
- There is `memory_reserved()`: How much GPU mem fenced off for this process. (might or might not be in use)
- There is `memory_allocated()`: How much GPU mem consumed by the objects stored on GPU (e.g tensors)

`memory_reserved` exists so the process and quickly put/remove thing from GPU without have to go through standard slow memory allocation

#### Tips
- [end](https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/3) of each step of the training loop should `del loss, output`
- if tracking metrics (e.g stats per batch) make sure to `detach()` so don't need full grad tree.

#### Details:
- A process gets GPU memory reserved for it. This is what `nivida-smi` sees. Happens austomatically shift tensor to GPU.
    - Can see the size of this with `cuda.memory_reserved()`
- The actual amoung of memory taken up (within the revered memory) is called `cuda.memory_allocated`.
- If you delete a tensor on gpu (or remove all reference to it)
    - `cuda.memory_allocated` will go down
    - `cuda.memory_reserved()` will not change.
- if you create a new tensor (of smaller size)
    - `cuda.memory_allocated` will go up
    - `cuda.memory_reserved()` will not change.
- if you create a new tensor (of smaller size)
    - `cuda.memory_allocated` will go up.
    - `cuda.memory_reserved()` will go up a little

There is some small memory overhead when you import packages (8.12mb). Doesn't appear to change with size of data moved to GPU.


##### Cleaning up memory:
deallocated from `memory_allocated()`
```python
del some_variable_on_gpu
# OR.  note cpu() makes a copy, doesn't delete, but if there is no other reference to the `some_variable_on_gpu` on mem, the gpu mem is freed
some_variable_on_gpu = some_variable_on_gpu.cpu()
```

deallocated from `memory_reserved`
`cuda.empty_cache()`
	- This can only free memory not required used in `memory_allocated()` (e,g stuff no longer referenced)
		- NOTE: might still be referenced if its in some[ computational graph](https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879/12) (see 2 comments down))

#### Tools
`nvidia-smi`: this give overview of the GPUs, and the **Allocated memory**

#### reference:

### Profiling GPU/Pytorch.
#### Overview:

| Name                                                                                            | pros                                                   | Weakness                                                                                                          | When to use                                        |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| CPyton (snakeviz)                                                                               | fast and easy, no changes to code                      | [Doesn't capture](https://discuss.pytorch.org/t/how-to-load-all-data-into-gpu-for-training/27609/8) GPU behaviour | CPU operation are the main issue in your program   |
| [torch.utils.bottleneck](https://pytorch.org/docs/stable/bottleneck.html)                       | Shows batch a Cprofile and a Cuda profile. Easy to run | Don't get easy to understand view of what called what                                                             | Use when need to work out where time is being used |
| [torch.cuda snapshot](https://pytorch.org/docs/stable/torch_cuda_memory.html#torch-cuda-memory) | easy to make nice visualisation of the mem             | Only work on linux                                                                                                | if have linus                                      |
| Nvidia Nsight                                                                                   | Give super detailed debugging information              | More complicated to use and interoperate                                                                          | Deep understanding                                 |
**Suggested flow:**
1) Use torch.utils.bottleneck to see how much is CPU GPU
	1) If CPU high: Us snakeviz for faster understanding. Need to be somewhat careful and the gpu distorts things (check notes below
	2) If you need to deeply understand the interaction/memory etc between CPU and GPU use Nsights

#### cPython/snakeviz
Very good starting place if need to your task is cpu workload. (cpu bound). Note, the following pitfalls of using it alongside GPU:
- Reports time for CPU to launch GPU kernal, but NOT GPU execution. (unless force to execute sychronously).
	- So might not see gpu at all.
- Ops that wait on GPU (sychronous) will appear extremely expensive (becuase see all the GPU time here)
more details [here](https://pytorch.org/docs/stable/bottleneck.html#module-torch.utils.bottleneck)

#### torch.utils.bottleneck
#Todo: read up on how to interperate or better make use of this
Sample output below: Note, it just reports the low level functions which is not very helpful

```text

	--------------------------------------------------------------------------------
	  cProfile output
	--------------------------------------------------------------------------------
	         270 function calls (269 primitive calls) in 4.448 seconds

	   Ordered by: internal time
	   List reduced from 113 to 15 due to restriction <15>

	   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
	        1    3.001    3.001    3.001    3.001 {built-in method time.sleep}
	        1    1.394    1.394    1.394    1.394 {built-in method torch._C._cudart.cudaProfilerStart}
	        1    0.047    0.047    0.047    0.047 {built-in method torch.rand}
	        3    0.002    0.001    0.002    0.001 {built-in method nt.stat}
	        1    0.002    0.002    0.002    0.002 {method 'cpu' of 'torch._C.TensorBase' objects}
	        1    0.001    0.001    0.001    0.001 {built-in method io.open_code}
	        1    0.001    0.001    0.001    0.001 {built-in method builtins.print}
	        1    0.000    0.000    0.000    0.000 {method 'pow' of 'torch._C.TensorBase' objects}
	        1    0.000    0.000    4.448    4.448 nsys_test.py:1(<module>)
	        1    0.000    0.000    0.000    0.000 {built-in method torch._C._nvtx.rangePushA}
	        1    0.000    0.000    0.000    0.000 {method 'read' of '_io.BufferedReader' objects}
	        1    0.000    0.000    0.000    0.000 {method '__exit__' of '_io._IOBase' objects}
	        5    0.000    0.000    0.000    0.000 <frozen importlib._bootstrap_external>:96(_path_join)
	        1    0.000    0.000    0.003    0.003 <frozen importlib._bootstrap>:1054(_find_spec)
	        1    0.000    0.000    0.002    0.002 <frozen importlib._bootstrap_external>:1604(find_spec)


	--------------------------------------------------------------------------------
	  autograd profiler output (CPU mode)
	--------------------------------------------------------------------------------
	        top 15 events sorted by cpu_time_total

	-----------------------  ------------  ------------  ------------  ------------  ------------  ------------
	                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
	-----------------------  ------------  ------------  ------------  ------------  ------------  ------------
	               aten::to         0.41%       8.000us        89.16%       1.735ms       1.735ms             1
	         aten::_to_copy         1.64%      32.000us        88.75%       1.727ms       1.727ms             1
	            aten::copy_        86.23%       1.678ms        86.23%       1.678ms       1.678ms             1
	             aten::rand         1.23%      24.000us         7.61%     148.000us     148.000us             1
	         aten::uniform_         3.55%      69.000us         3.55%      69.000us      69.000us             1
	              aten::pow         3.08%      60.000us         3.24%      63.000us      63.000us             1
	            aten::empty         2.83%      55.000us         2.83%      55.000us      55.000us             1
	    aten::empty_strided         0.87%      17.000us         0.87%      17.000us      17.000us             1
	      aten::result_type         0.10%       2.000us         0.10%       2.000us       2.000us             1
	               aten::to         0.05%       1.000us         0.05%       1.000us       1.000us             1
	-----------------------  ------------  ------------  ------------  ------------  ------------  ------------
	Self CPU time total: 1.946ms

	--------------------------------------------------------------------------------
	  autograd profiler output (CUDA mode)
	--------------------------------------------------------------------------------
	        top 15 events sorted by cpu_time_total

	        Because the autograd profiler uses the CUDA event API,
	        the CUDA time column reports approximately max(cuda_time, cpu_time).
	        Please ignore this output if your code does not use CUDA.

	-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
	                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
	-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
	               aten::to         0.33%      19.000us        90.40%       5.178ms       5.178ms       4.000us         0.07%       5.281ms       5.281ms             1
	         aten::_to_copy         1.03%      59.000us        90.07%       5.159ms       5.159ms       6.000us         0.11%       5.277ms       5.277ms             1
	            aten::copy_        86.38%       4.948ms        86.38%       4.948ms       4.948ms       5.269ms        97.25%       5.269ms       5.269ms             1
	             aten::rand         1.75%     100.000us         6.98%     400.000us     400.000us       7.000us         0.13%      49.000us      49.000us             1
	         aten::uniform_         4.17%     239.000us         4.17%     239.000us     239.000us      40.000us         0.74%      40.000us      40.000us             1
	    aten::empty_strided         2.65%     152.000us         2.65%     152.000us     152.000us       2.000us         0.04%       2.000us       2.000us             1
	              aten::pow         2.51%     144.000us         2.62%     150.000us     150.000us      84.000us         1.55%      88.000us      88.000us             1
	            aten::empty         1.06%      61.000us         1.06%      61.000us      61.000us       2.000us         0.04%       2.000us       2.000us             1
	      aten::result_type         0.07%       4.000us         0.07%       4.000us       4.000us       2.000us         0.04%       2.000us       2.000us             1
	               aten::to         0.03%       2.000us         0.03%       2.000us       2.000us       2.000us         0.04%       2.000us       2.000us             1
	-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
	Self CPU time total: 5.728ms
	Self CUDA time total: 5.418ms
```

#### [[Nvidia Nsight system]]
- See related article/

#### Guide:
- [sagivtech](https://www.sagivtech.com/2017/09/19/optimizing-pytorch-training-code/): rought highlevel guide and ddataloader to get quick wins. not detailed.


---
# Reference
