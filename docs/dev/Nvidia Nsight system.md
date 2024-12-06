---
time: 2024-07-12 08:52
tags:
  - pytorch
---
# Nvidia Nsight system

## Overview
Detailed profiling to gpu/gpu etc. https://developer.nvidia.com/nsight-systems
## Details

### Collecting results - Option 1: Command line
pros: Track only the relevant period. Give you granular control over when you record
cons: Tricker to set up:

#### Steps:
##### 1) Open `powershell` as admin
1) Find in the start menu, right click, run as administrator
Do this because you need the right permission to trace cpu. There is probably other ways of getting this through bash etc.
##### 2) Install/find nsys
1) Easiest: find the Nsight system install on machine, run it by directly pointing to it
	- e.g my powershell path `& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.1\target-windows-x64\nsys"`
	- to check it works run: `& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.1\target-windows-x64\nsys" --help`
2) Install nsys in your local environment. (I was unsuccessful):
	1) the following might provide some help for install in .venv
		1) `C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.1\target-windows-x64\python\packages\nsys_recipeinstall.py`
		2) Tools is detailed [here](https://docs.nvidia.com/nsight-systems/2023.3/pdf/InstallationGuide.pdf).
##### 3) Call the right commands
(note replace nsys with the full path to nsys if need by):
- Basic command to get proof of life
	- `nsys profile python python_script.py`
		- `python`:  path to any python install (e.g venv is fine). Need to have the dependancies for the script.
		-  `python_script.py`: the script to execute
	- This will output a fill to the cwd of the terminal the script was called from
###### Useful tracking:
Following uses the `NVTX` library. I had trouble install but `torch` makes it available ([example](https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59)).
NOTE: for function tracing, typically a good idea to "warm up" the function as the first run has overheads. See [here](https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59)

- Adding specific start and end points for recording:
	- `-c cudaProfilerApi`:  Tells it to only start recording when it hit `cudaProfilerStart()` in code.
		- e.g `torch.cuda.cudart().cudaProfilerStart()`
	- `--capture-range-end=stop`: Tells it when to stop recording.
		- `stop`: when it hits `torch.cuda.cudart().cudaProfilerStop()` in code.
		- Other useful options, like once block has executed enough times.
- Tracing specific function:
	- You can write messages to the "NVTX" row in the GUI by adding the following in your code.
	- Note: this can be nested, which will give you an icicle appearance (like snakeviz).
	```python
	torch.cuda.nvtx.range_push("msg_to_call_block")
	# SOME CODE/FUNCTION whole executation should be recorded in block "msg_to_call_block"
	torch.cuda.nvtx.range_pop() # Marks this block is over, will pop and retrun to the next block
	```
	- use `nvtx` pakcage:
		- I couldn't get install to work, but the git is [here](https://github.com/NVIDIA/NVTX/tree/release-v3/python)
Trace functions without code change:
- `--python-functions-trace=path_to_trace_file_config.json`
- `path_to_trace_file_config.json` looks like this (example pointed to in `nsys profile --help`)

	```json
	[
		{
			"_comment": "Mandatory: Name put on the block in GUI",
			"domain": "(Optional) My Domain - this adds a row like the NTVX row",
			"category": "(Optional) My Category- this is a sub row under that",
			"color": "(Optionsal) 0x008000",
			"module": "torch.utils.dummy_func",
			"functions": ["local_func"]
		},
		{...}
	]
	```
	- Will produce something like this in the GUI:
		- ![[Pasted image 20240712084854.png|300]]
	- NOTE: you can't trace any function defined the file you are running (e.g `__main__`)

For explanation of different command line arguments see [here](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#openacc-trace)

#### full examples command:
`& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.1\target-windows-x64\nsys" profile -f true  -c cudaProfilerApi  --capture-range-end=stop --python-functions-trace=C:\Users\SEBWIN\Documents\technical\code\TDR_axtreme\nsight_annotations.json  C:\Users\SEBWIN\Documents\technical\code\TDR_axtreme\.venv\Scripts\python.exe C:\Users\SEBWIN\Documents\technical\code\TDR_axtreme\nsys_test.py`


#### Trouble shooting:

If you don't get a message like this in the terminal at the end it didn't run properly:
```
Generating 'C:\Users\SEBWIN\AppData\Local\Temp\nsys-report-5cc6.qdstrm'
[1/1] [========================100%] report3.nsys-rep
Generated:
    C:\Users\SEBWIN\Documents\technical\code\TDR_axtreme\report3.nsys-rep
```

### Collecting results -  Option 2: GUI
in "command line with argument": `-w true -t cuda,nvtx -s cpu  --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true -x true -o my_profile C:\Users\SEBWIN\Documents\technical\code\TDR_axtreme\.venv\Scripts\python.exe C:\Users\SEBWIN\Documents\technical\code\TDR_axtreme\junk.py`

### Understanding the results:
- Need to `nttx` or `--python-functions-trace=<path_to_profiling_json>` to link parts of script to profiling activity (otherwise its basically impossible)
- For a specific row you are interest in "show events view", then click the specifc event to navigate to is
	- ![[Pasted image 20240712081022.png|200]]

## References:
[Paulbridger](https://paulbridger.com/posts/nsight-systems-systematic-optimization/):  Slightly out of data overview for how to use/undstand Nivida Nsight
[ptrblack guide on Nvida Nsight for pytorch](https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59): Slightly out of data, very useful to see how to call NVTX from torch.
[Nvidia intro series](https://www.youtube.com/watch?v=xdFQZSV5IrU&list=PL5B692fm6--ukF8S7ul5NmceZhXLRv_lR): youtube explain overview of how to use.
Nvidea material:
- [Deep slide pack on understanding output](https://alcf.anl.gov/sites/default/files/2022-07/Nsight%20Systems%20-%20DL%20Profiling%20Argonne%20National%20Labs%202022-06-30.pdf): incredible resource
-  [related](https://indico.cern.ch/event/962112/contributions/4047370/attachments/2159916/3643963/Nsight%20Systems%20-%20x86%20Introduction%20-%20CERN.pdf)
- https://www.vi-hps.org/cms/upload/material/tw41/Nsight_Systems.pdf



---
# Reference
