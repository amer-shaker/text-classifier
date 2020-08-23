package com.mach.learn;

import java.io.Serializable;
import java.util.Collection;

/**
 * A basic wrapper reflecting a classification. It will store both features
 * and resulting classification.
 *
 * @param <T> The feature class
 * @param <K> The category class
 * @author Amer Shaker
 */
public class Classification<T, K> implements Serializable {

    /**
     * Generated Serial Version UID (generated for 1.0.0.RELEASE).
     */
    private static final long serialVersionUID = -1210981535415341283L;

    /**
     * The classified features.
     */
    private final Collection<T> features;

    /**
     * The category as which the features was classified.
     */
    private final K category;

    /**
     * The probability that the features belongs to the given category.
     */
    private final double probability;

    /**
     * Constructs a new Classification with the parameters given and a default
     * probability of 1.
     *
     * @param features The features.
     * @param category The category.
     */
    public Classification(Collection<T> features, K category) {
        this(features, category, 1);
    }

    /**
     * Constructs a new Classification with the parameters given.
     *
     * @param features    The features.
     * @param category    The category.
     * @param probability The probability.
     */
    public Classification(Collection<T> features, K category, double probability) {
        this.features = features;
        this.category = category;
        this.probability = probability;
    }

    /**
     * Retrieves the features classified.
     *
     * @return The features.
     */
    public Collection<T> getFeatures() {
        return features;
    }

    /**
     * Retrieves the category the features was classified as.
     *
     * @return The category.
     */
    public K getCategory() {
        return category;
    }

    /**
     * Retrieves the classification's probability.
     *
     * @return the probability.
     */
    public double getProbability() {
        return probability;
    }

    @Override
    public String toString() {
        return "Classification{" +
                "features=" + features +
                ", category=" + category +
                ", probability=" + probability +
                '}';
    }
}
