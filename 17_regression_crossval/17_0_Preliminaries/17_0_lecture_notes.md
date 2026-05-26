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

To solve this, we can compute a new metric called **Variance**, which takes the TSS and divides it by the size of the sample. The variance is simply the average of the squared deviations. For Sample A, the variance is 0. For Sample B, the variance is 36 ($7,200 / 200 == 36$), which is the exact same variance we would have calculated with our original sample of 100 ($3,600 / 100 = 36$). 

We can take this simplification one step further. Because variance is measured in squared units (squared inches), we can take the square root of the variance to put our metric back into our original unit of measure. This metric is called the **Standard Deviation**. For Sample B, the variance is 36 squared inches; the square root of 36 is 6 inches. This tells us that, on average, students in Sample B deviate from the mean by 6 inches. 

Now we have the mean plus three new metrics to assess the spread of our data: Total Sum of Squares (TSS), Variance, and Standard Deviation.

> **Note:** When we calculated the variance, we divided the TSS by the total number of data points ($N$). In practice, you will often see the variance calculated by dividing the TSS by the total number of data points minus one ($N-1$). Just using $N$ produces a variance that tends to underestimate the actual population variance. Since we will almost always be working with samples and won't have complete population data, $N-1$ is most commonly used as the divisor to better estimate the population variance. I am not going to go further into the details of why this works in this class. If you want to know more details about why this is done, [read here about Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction). For now, When you see $N$ or $N-1$ in the divisor, I just want you to think of the same basic concept: we are dividing by the size of the data to find an average. In the case of covariance, we a just getting the average squared deviation). 


---
## Part 2: Covariance - Correlation

Let's start over. Same goal, we're going to estimate an unseen student's height. We have data on the heights of 100 other students from the same school. But this time, we have some additional information. We know the shoesize of the unseen student, and we know both the height and the shoesize of the 100 other students. 

In the last part, we couldn't do better than guessing the mean and examining the spread of the height data to get a sense of how wrong me might possibly be. Having the unseen student's shoe size changes the game for us. We might expect that larger feet are generally attached to taller people, and that smaller feet are generally attached to smaller people. Since we know our unseen student's show size, we may be able to do better than guess the mean. 

But first we deed to confirm are assumptions that feet size and height tend go together. Ideally, we could calculate a single metric that would allow us to characterize the relation between the two variables. We can do this in a way that is very similar to how we calculated variance in the previous lecture. However, instead of just calculating variance for height, we are going to calculate variance for both height and shoe size. 

To breifly recap caclculating variance: we will take one of variables, shoe size in this case, we will make a list of all the shoe sizes of the students in our sample. We will then subtract the mean shoe size from those values and produce a list of deviations. We will do the same thing for height and we will wind up with a deviation for height and a deviation for show size for each student. 

Now, imagine we plot our original height and shoe size with height on the x axis and show size on the y axis. We then center our plot so that the mean values for height and show size are dead center. A student who is perfectly average in both regards would be plotted in the center. Splitting the graph this way would create four quadrants that we could use to classify our students: 

* Taller than average, Bigger feet than average: In the top right quadrant. These student will a positve deviation for both height and shoe size. 

* Shorter than average, Smaller feet than average: In the bottom left quadrant. These students will have a negative deviation for both height and shoe size. 

* Taller than average, Smaller feet than average: In the bottom right quadrant. These students will have a positive deviation for height and a negative deviation for shoe size. 

* Shorter than average, Bigger feet than average: In the top left quadrant. These student will have a negative deviation for height and a positive deviation for shoe size. 

If we look at all of the students plotted on this graph, we would start to get a a sense of whether or not we are right about taller people having larger feet. If we were right, what would we expect the data in these quadrants to look like? Just eye-balling it, it seems like we should expect more students in the Tall/Big and the Short/Small quadrants. However, what would it mean if we counted up the number of students in each quadrant and found they were equal? If number of Taller/Smaller + Shorter/Bigger is roughly equal to the Taller/Bigger + Shorter/Smaller count we might start to worry about our hypothesis. 

We can do a bit more calculation to actually settle the issue. If we muliply the deviation of height by the deviation of shoesize for each participant we will end up with positive values for students in the Taller/Bigger quadrant and the Shorter/Smaller quadrant. In contrast, we wind up with negative values for students in the Taller/Smaller and Smaller/Bigger quadrants. 

If we then take these values and add them altogether we get a single value. As we discussed previously, metrics that increase as our sample size increases are often going to cause us problems. So we take the sum of the multiplied deviations and we divide it by the sample size. We are getting the average of the sum of the multiple deviation. This is a metric we call the **Covariance**. 

If the covariance is positive, this indicates more students are following the Taller/Bigger + Shorter/Smaller trend. If the covariance is negative, we are seeing more of a Taller/Smaller + Shorter/Bigger trend. Finally if the covariance is close to zero, then all of the quadrants are about equal and we aren't seeing any real trend. 

Once we do this, our covariacne is expressed in a number in the metric of both the x and the y axis. For this particular example, the units of measurement for out covariance would be inches * barleycorns ([the unit used in American shoe sizes](https://en.wikipedia.org/wiki/Barleycorn_(unit))). 

Now, inches * barleycorns is not a particularly helpful unit of measure so we need to do some conversion here. Our strategy is going to be to make our metric unitless. To do so, we are going to take the standard deviation of height (which is in inches) and multiply that by the standard deviation of shoe size (which is in barelycorns). This produces a number that is in the inches * barleycorn metric. 

We then take our covariance and divide it by this new number constructed by multiplying the standard deviations of our two variables. The result is a unitless value, that we call a correlation coefficient. More precisely, we have generated Pearsons' correlation, often abbreviated as `r`.    