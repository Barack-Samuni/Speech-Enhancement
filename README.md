# Speech-Enhancement
In This repository, we will aim to improve the speech enhancement capabilities for our team.
We will explore and hopefully implement some new speech enhancement methods.
Optional: Create a python library out of this project that can be installed via pip :)

## Setting up an environment
To set up an environment, write the following commands:

```
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
### Note:
It makes absolute sense that the requirements 
between methods might defer,because each method might 
use different libraries.
So, it would be better if each member of the team 
who's working on a method created a 
requirements file of his own, 
which will be merged eventually to a single file.

### Creating a requirements file
After installing your desired libraries, simply write the following command (After activating your environment):
```
pip freeze>requirements.txt
```

## Datasets and outputs handling
- All Datasets will be saved in a folder called [datasets](datasets/)
- All Outputs will be saved in a folder called [outputs](outoputs/)

**This folders will not be Uploaded to git!**

**Good luck to all of us! :)**