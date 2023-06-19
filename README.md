This repository contains functions for analysis of in vivo electrophysiology data. It is still in early stages of development, but all functions in the main branch should be functional. 

This goal of this module is to be very flexible, allowing for use with many different file types. To do this, all the functions run using timestamps and values associated to those timestamps. In the future, integration with Neo/NWB would be ideal.

Current installation instructions.
-These instructions assumes that user is using the Anaconda python distribution
Clone repository. There are two options:

1. Using git (all branches). In git bash type:

`git clone https://github.com/AlexLM96/invian.git`

2. Downloading .zip file:

In invian repository (https://github.com/AlexLM96/invian/tree/dev) click `code` and then `download zip`

Then open Anaconda prompt and type:

`pip install INVIAN_LOCATION`

where INVIAN_LOCATION is the file path of the invian folder. Note that the path should point to the main folder. E.g: `pip install C:\Users\alex\Desktop\invian`