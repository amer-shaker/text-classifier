package com.mach.learn;

/**
 * Simple interface defining the method to calculate the feature probability.
 *
 * @param <T> The feature class
 * @param <K> The category class
 * @author Amer Shaker
 */
public interface IFeatureProbability<T, K> {

    /**
     * Returns the probability of a <code>feature</code> being classified as
     * <code>category</code> in the learning set.
     *
     * @param feature  the feature to return the probability for
     * @param category the category to check the feature against
     * @return the probability <code>P(feature|category)</code>
     */
    double featureProbability(T feature, K category);
}
