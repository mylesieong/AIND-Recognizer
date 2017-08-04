# Setup hmmlearn library Probelm [SOLVED]
When we enter part 2, ModuleNotFoundError prompt from `from hmmlearn.hmm import GaussianHMM` statement. Use pip install hmmlearn cannot install hmmlearn (saying it needs Visual C++ 2015 dependency but actually its available already). The solution is that we open Anaconda prompt and run `pip install hmmlearn‑0.2.1‑cp36‑cp36m‑win32.whl` (this file is donwloaded from [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/) thanks to the idea given at stackoverflow [question](https://stackoverflow.com/questions/42468700/why-hmmlearn-cant-install-in-anaconda3-prompt)). 
I learnt these during the problem shooting:
1. In Jupyter, in the kernel dropdown list there is an option that runs all script until finish or error prompt. Very useful
2. conda-env list
3. conda list/ pip list
4. In Anaconda, environments are arranged as below: ($ANA_HOME = /cygdrive/c/ProgramData/Anaconda3)
    * Default root:  $ANA_HOME/
    * Default pylib: $ANA_HOME/Lib + $ANA_HOME/Lib/site-packages
    * Env root:      $ANA_HOME/envs/$ENV_NAME
    * Env pylib:     $ANA_HOME/envs/$ENV_NAME/Lib + $ANA_HOME/envs/$ENV_NAME/Lib/site-packages 
5. God helps ones who help themselves
