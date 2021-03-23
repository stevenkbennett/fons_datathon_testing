# FONS Datathon
Automated checking repository

This is a repository for automated testing of your predictions against the test data.
To score your predictions, simply navigate to your team's branch and upload your predictions as a `.csv` file using the names `task_x_predictions.csv` or `bonus_x_predictions.csv` according to the number of task you wish to be scored. Then, if not already done so, create a pull request from your branch into the main branch. Once you have submitted your pull request, wait a few minutes and your results will appear in the comments! If you have already created a pull request, all you have to do is navigate to it and again, wait a few minutes for your scores. 

You do not have to submit all the tasks at once! Feel free to submit as many of tasks or bonus tasks as you wish! 


These `.csv` files should be one prediction per line, with a total line length of 3,363 to match the test dataset size.
If you do not submit this, the scoring will fail.
The script used to score your models is `test_scoring.py` so check that file if you're confused as to why you're scoring doesn't work.


If you are confused how to format your `.csv`, see `example_predictions.csv`, the `Getting_started.ipynb` notebook or ask a demonstrator.
If you are new to GitHub and haven't submitted a pull request before, please ask a demonstrator.
An example pull request can be seen by selecting the `pull requests` tab on the top and looking in `Example Pull Request`.
