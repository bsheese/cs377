## Part 1: Prediction from Mean, Deviation, Total Sum of Squares, Variance, Standard Deviation

Let's say we want to make a prediction about the height of a student we haven't met yet. We don't know anything about the student, but we do know the heights of 100 students who go to the same school. In this situation, our best bet for guessing the height of an unknown student is to calculate the average height of the 100 students and guess that average.

We may be way off. The student might be considerably taller or considerably shorter than the 100-student average. What we've really done when we guess the average is not say, "I think this student of unseen height will be exactly this height." Instead, what we are saying is something like: "I have no idea what the height of this student will be. But, if I guess the mean, I can minimize how wrong I might be in either direction."

Two things to note here:
* Guessing the mean is going to be our baseline model. All the other techniques we're going to cover are basically attempts to do better than this baseline model. In noisy, real-world data, sometimes we will fail to do better.
* It is useful to think of the tools we will cover as attempts to minimize error rather than to divine an exact prediction. A good model will give us confidence that our predictions fall within some range. Bad models aren't much better than flipping a coin. There will be some element of chance in everything we do, but we want to be more like forecasters than gamblers.

So let's go back to our height prediction problem. Imagine we have two distinct samples of 100 students' heights. To keep the math clean, we will measure everyone in **inches**.

* For **Sample A**, we measured 100 students, and every single one of them was exactly 66 inches tall (five and a half feet).
* For **Sample B**, we measured 100 students; 50 of them were 60 inches tall (five feet) and the other 50 were 72 inches tall (six feet).

If we followed our basic model of predicting the mean, both samples would lead us to predict 66 inches, which is the average for both samples. However, if we pay attention to the spread of our data (its distribution), we might have much less confidence guessing the mean for the second sample. Notably, in Sample B, we've observed zero students who are actually 66 inches tall. We might feel like guessing the mean is almost certainly going to be wrong.

It would be useful to have a metric to help us differentiate between the spread of these two samples. One thing we could do is calculate the difference between each observed student and the mean of the sample. We do this by subtracting the mean from each individual value. This produces a list of differences called **deviations**.

If we want to turn our list of deviations into a single metric, we might be tempted to just add them all up. The sum of the deviations for Sample A is 0, since none of the students differ from the mean. Unfortunately, the sum of the deviations for Sample B is also 0. Because the mean is 66 inches, half of our deviations are -6 (60 minus 66) and the other half are +6 (72 minus 66). When we add fifty -6s and fifty +6s together, they cancel out to 0. This isn't a fluke. The mean is always the exact value that perfectly balances the deviations in any dataset.

To avoid this balancing problem, we could either take the absolute value of the deviations or square them. Both methods make all the numbers positive, solving the cancellation problem, but squaring has the additional effect of disproportionately penalizing larger errors. A deviation of 10 squared becomes 100, while a deviation of 100 squared becomes 10,000. This additional penalty for larger errors is often preferred in statistics, so summing the squared deviations is the standard approach.

If we take all of our deviations, square each one individually, and then sum them all up, we get a single metric called the **Total Sum of Squares (TSS)**. This is our first metric for comparing the spread of data in different samples.
* For Sample A, the TSS is 0 because all the deviations are 0.
* For Sample B, we take each deviation (6), square it (36), and sum them across all 100 students ($36 \times 100$), giving us a TSS of 3,600 squared inches.

Now let's say we took Sample A and Sample B and doubled the sample size of each. Sample A now consists of 200 students who are all exactly 66 inches tall. Sample B consists of 200 students, half of whom are 60 inches and half of whom are 72 inches. The means for both samples do not change, but the TSS does. Sample A's TSS is still 0, but Sample B's TSS doubles to 7,200 ($36 \times 200$). This property of TSS, where simply adding more data points increases the value, isn't ideal if we only want to measure and compare the general spread of our data.

To solve this, we can compute a new metric called **Variance**, which takes the TSS and divides it by the size of the sample. The variance is simply the average of the squared deviations. For Sample A, the variance is 0. For Sample B, the variance is 36 ($7,200 / 200 = 36$), which is the exact same variance we would have calculated with our original sample of 100 ($3,600 / 100 = 36$).

We can take this simplification one step further. Because variance is measured in squared units (squared inches), we can take the square root of the variance to put our metric back into our original unit of measure. This metric is called the **Standard Deviation**. For Sample B, the variance is 36 squared inches; the square root of 36 is 6 inches. This tells us that, on average, students in Sample B deviate from the mean by 6 inches.

Now we have the mean plus three new metrics to assess the spread of our data: Total Sum of Squares (TSS), Variance, and Standard Deviation.

> **Note:** When we calculated the variance, we divided the TSS by the total number of data points ($N$). In practice, you will often see the variance calculated by dividing the TSS by the total number of data points minus one ($N-1$). Just using $N$ produces a variance that tends to underestimate the actual population variance. Since we will almost always be working with samples and won't have complete population data, $N-1$ is most commonly used as the divisor to better estimate the population variance. I am not going to go further into the details of why this works in this class. If you want to know more details about why this is done, [read here about Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction). For now, when you see $N$ or $N-1$ in the divisor, I just want you to think of the same basic concept: we are dividing by the size of the data to find an average. In the case of variance, we are just getting the average squared deviation.


---
## Part 2: Covariance - Correlation

Let's start over. Same goal: we're going to estimate an unseen student's height. We have data on the heights of 100 other students from the same school. But this time, we have some additional information. We know the shoe size of the unseen student, and we know both the height and the shoe size of the 100 other students.

In the last part, we couldn't do better than guessing the mean and examining the spread of the height data to get a sense of how wrong we might possibly be. Having the unseen student's shoe size changes the game for us. We might expect that larger feet are generally attached to taller people, and that smaller feet are generally attached to shorter people. Since we know our unseen student's shoe size, we may be able to do better than guess the mean.

But first we need to confirm our assumption that foot size and height tend to go together. Ideally, we could calculate a single metric that would allow us to characterize the relation between the two variables. We can do this in a way that is very similar to how we calculated variance in the previous lecture. However, instead of just calculating variance for height, we are going to work with the deviations of both height and shoe size.

To briefly recap calculating deviations: we will take one of our variables, shoe size in this case, and make a list of all the shoe sizes of the students in our sample. We will then subtract the mean shoe size from those values and produce a list of deviations. We will do the same thing for height, and we will wind up with a deviation for height and a deviation for shoe size for each student.

Now, imagine we plot our original height and shoe size data with height on the y-axis and shoe size on the x-axis. We then center our plot so that the mean values for height and shoe size are dead center. Splitting the plot this way creates four quadrants that we could use to classify our students:

* Taller than average, bigger feet than average: in the top right quadrant. These students will have a positive deviation for both height and shoe size.

* Shorter than average, smaller feet than average: in the bottom left quadrant. These students will have a negative deviation for both height and shoe size.

* Taller than average, smaller feet than average: in the top left quadrant. These students will have a positive deviation for height and a negative deviation for shoe size.

* Shorter than average, bigger feet than average: in the bottom right quadrant. These students will have a negative deviation for height and a positive deviation for shoe size.

If we look at all of the students plotted on this graph, we would start to get a sense of whether or not we are correct about taller people having larger feet. If we were correct, what would we expect the data in these quadrants to look like? Just eye-balling it, it seems like we should expect more students in the Tall/Big and the Short/Small quadrants. However, what would it mean if we counted up the number of students in each quadrant and found they were equal? If the Taller/Smaller + Shorter/Bigger count is roughly equal to the Taller/Bigger + Shorter/Smaller count, we might start to worry about our hypothesis.

We can do a bit more calculation to actually settle the issue. If we multiply the deviation of height by the deviation of shoe size for each student, we will end up with positive values for students in the Taller/Bigger quadrant and the Shorter/Smaller quadrant (a positive times a positive, or a negative times a negative). In contrast, we wind up with negative values for students in the Taller/Smaller and Shorter/Bigger quadrants (one positive deviation times one negative deviation).

If we then take these values and add them all together, we get a single value: the Sum of the Cross-Products. As we discussed previously, metrics that increase as our sample size increases are often going to cause us problems. So we go a bit further: we take the sum of the multiplied deviations and divide it by the sample size (or $N-1$; see the note above on Bessel's correction). We are getting the average of the cross-products. This is a metric we call the **Covariance**.

If the covariance is positive, this indicates more students are following the Taller/Bigger + Shorter/Smaller trend. If the covariance is negative, we are seeing more of a Taller/Smaller + Shorter/Bigger trend. Finally, if the covariance is close to zero, then all of the quadrants are about equal and we aren't seeing any real trend.

Once we do this, our covariance is expressed in a number in the units of both the x and the y axis. For this particular example, the units of measurement for covariance would be inches × barleycorns ([the unit used in American shoe sizes](https://en.wikipedia.org/wiki/Barleycorn_(unit))).

Now, inches × barleycorns is not a particularly helpful unit of measure, so we need to do some conversion here. Our strategy is going to be to make our metric unitless. To do so, we are going to take the standard deviation of height (which is in inches) and multiply that by the standard deviation of shoe size (which is in barleycorns). This produces a number that is in the inches × barleycorns metric.

We then take our covariance and divide it by this new number constructed by multiplying the standard deviations of our two variables. The result is a unitless value that we call a correlation coefficient. More precisely, we have generated Pearson's correlation coefficient, often abbreviated as `r`.

---
## Part 3: Residuals

So we now know how to calculate Pearson's r: a unitless metric that indicates the direction and strength of an association. We are now going to shift from describing our data to making predictions. By predictions, I mean that we are going to get a new x value, and we will develop a model that allows us to estimate y for that value of x. We will come back around to r in just a moment; it will play a role in how we make predictions.

So let's go back to our height prediction problem. We have a bunch of students and know their height and their shoe size. We have an unseen student and we know their shoe size; our job is to predict their height. How are we going to get a specific prediction for height for any given value of shoe size? We want to move beyond just predicting the mean of height every time. We want to incorporate the additional information from the shoe size. The simplest way to do that is to generate a straight line through our data. We can describe any straight line we can imagine by describing just two things: the slope of the line (how much y changes for every unit increase in x) and the intercept (where the line crosses the y-axis).

Imagine you are trying to draw a line through a scatterplot of our data. If the job of the line is to capture the relation between x and y, you want a line that follows the shape of the association. This is easy if the data all line up in something that looks exactly like a line. But, for the data we will use in this class, we will almost never see something like that. Instead we will get data with quite a bit of scatter; rather than clear lines we get clouds that only sometimes have discernible trends in one direction or another. So, let's say we've got a cloud and a ruler and we draw a line on our plot, but after looking at it, it doesn't seem right, so we try again. Now we've made a second line on our plot, and maybe it's better, but we're not sure. We need to calculate how wrong these two lines are so we can see which one is better.

To do this we are going to look at the difference between what the line predicted for y at each value of x and the actual y value. This may sound familiar: earlier we looked at the difference between the mean and an actual value, which we called the deviation. We're doing the same thing here, but now we are looking at the difference between the actual y data point and our line's prediction for it. This difference is called the residual. Statisticians call the actual value y, and they call the predicted value y-hat. A residual is calculated for each data point by taking y and subtracting y-hat (actual minus predicted). We wind up producing a list of residuals. These residuals can be positive, indicating the actual value sits above the line (the line under-predicted), negative, indicating the actual value sits below the line (the line over-predicted), or zero if the prediction is exactly on target.

We can create a list of residuals for our first line and a second list for our second line. If we simply sum up the residuals, we wind up with the same problem we had before with the deviations. That is, the positive and negative values can cancel each other out and obscure how much error our line produces. However, it is actually a little bit worse this time: we can show that any line, no matter what slope, that crosses through the point at the mean of the y values and the mean of the x values will have perfectly balanced residuals that cancel out and produce zero.

We can solve these problems the same way we did before: we square the residuals first, and then add them together. This creates a single value indicating the incorrectness of our line, called the **Residual Sum of Squares (RSS)**. If we are just trying to compare our two lines, the line with the lower RSS is the winner — it is closer to fitting the pattern of the data than the other line.

However, we don't typically just draw lines and compare them. What we really want is the one line, out of all possible lines, with the lowest RSS: the line of best fit. It turns out there is a closed-form formula for that line, built from the pieces we already have — the optimal slope is $r \cdot (\sigma_y / \sigma_x)$, and the optimal intercept is whatever value makes the line pass through the point of means. How we get there, and how we score the resulting line with $R^2$, is the subject of notebook 17_0_5.
