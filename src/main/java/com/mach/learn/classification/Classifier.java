package com.mach.learn.classification;

import com.mach.learn.Classification;
import com.mach.learn.IFeatureProbability;

import java.io.Serializable;
import java.util.*;

/**
 * Abstract base extended by any concrete classifier. It implements the basic
 * functionality for storing categories or features and can be used to calculate
 * basic probabilities – both category and feature probabilities. The classify
 * function has to be implemented by the concrete classifier class.
 *
 * @param <T> A feature class
 * @param <K> A category class
 * @author Amer Shaker
 */
public abstract class Classifier<T, K> implements IFeatureProbability<T, K>, Serializable {

    /**
     * Generated Serial Version UID (generated for v1.0.0.RELEASE).
     */
    private static final long serialVersionUID = 5504911666956811966L;

    /**
     * Initial capacity of category dictionaries.
     */
    private static final int INITIAL_CATEGORY_DICTIONARY_CAPACITY = 16;

    /**
     * Initial capacity of feature dictionaries. It should be quite big, because
     * the features will quickly outnumber the categories.
     */
    private static final int INITIAL_FEATURE_DICTIONARY_CAPACITY = 32;

    /**
     * The initial memory capacity or how many classifications are memorized.
     */
    private int memoryCapacity = 1000;

    /**
     * A dictionary mapping features to their number of occurrences in each
     * known category.
     */
    private Map<K, Map<T, Integer>> featureCountPerCategory;

    /**
     * A dictionary mapping features to their number of occurrences.
     */
    private Map<T, Integer> totalFeatureCount;

    /**
     * A dictionary mapping categories to their number of occurrences.
     */
    private Map<K, Integer> totalCategoryCount;

    /**
     * The classifier's memory. It will forget old classifications as soon as
     * they become too old.
     */
    private Queue<Classification<T, K>> memoryQueue;

    /**
     * Constructs a new classifier without any trained knowledge.
     */
    public Classifier() {
        this.reset();
    }

    /**
     * Resets the <i>learned</i> feature and category counts.
     */
    public void reset() {
        this.featureCountPerCategory = new Hashtable<>(INITIAL_CATEGORY_DICTIONARY_CAPACITY);
        this.totalFeatureCount = new Hashtable<>(INITIAL_FEATURE_DICTIONARY_CAPACITY);
        this.totalCategoryCount = new Hashtable<>(INITIAL_CATEGORY_DICTIONARY_CAPACITY);
        this.memoryQueue = new LinkedList<>();
    }

    /**
     * Returns a <code>Set</code> of features the classifier knows about.
     *
     * @return The <code>Set</code> of features the classifier knows about.
     */
    public Set<T> getFeatures() {
        return totalFeatureCount.keySet();
    }

    /**
     * Returns a <code>Set</code> of categories the classifier knows about.
     *
     * @return The <code>Set</code> of categories the classifier knows about.
     */
    public Set<K> getCategories() {
        return (totalCategoryCount).keySet();
    }

    /**
     * Retrieves the total number of categories the classifier knows about.
     *
     * @return The total category count.
     */
    public int getCategoriesTotal() {
        return totalCategoryCount.size();
    }

    /**
     * Retrieves the memory's capacity.
     *
     * @return The memory's capacity.
     */
    public int getMemoryCapacity() {
        return memoryCapacity;
    }

    /**
     * Sets the memory's capacity. If the new value is less than the old value,
     * the memory will be truncated accordingly.
     *
     * @param memoryCapacity The new memory capacity.
     */
    public void setMemoryCapacity(int memoryCapacity) {
        for (int i = this.memoryCapacity; i > memoryCapacity; i--) {
            this.memoryQueue.poll();
        }
        this.memoryCapacity = memoryCapacity;
    }

    /**
     * Increments the count of a given feature in the given category. This is
     * equal to telling the classifier, that this feature has occurred in this
     * category.
     *
     * @param feature  The feature, which count to increase.
     * @param category The category the feature occurred in.
     */
    public void incrementFeature(T feature, K category) {
        Map<T, Integer> features = featureCountPerCategory.computeIfAbsent(category, k ->
                new Hashtable<>(INITIAL_FEATURE_DICTIONARY_CAPACITY));

        Integer count = Optional.ofNullable(features.get(feature)).orElse(0);
        features.put(feature, ++count);

        Integer totalCount = Optional.ofNullable(totalFeatureCount.get(feature)).orElse(0);
        totalFeatureCount.put(feature, ++totalCount);
    }

    /**
     * Increments the count of a given category. This is equal to telling the
     * classifier, that this category has occurred once more.
     *
     * @param category The category, which count to increase.
     */
    public void incrementCategory(K category) {
        Integer count = Optional.ofNullable(totalCategoryCount.get(category)).orElse(0);
        totalCategoryCount.put(category, ++count);
    }

    /**
     * Decrements the count of a given feature in the given category. This is
     * equal to telling the classifier that this feature was classified once in
     * the category.
     *
     * @param feature  The feature to decrement the count for.
     * @param category The category.
     */
    public void decrementFeature(T feature, K category) {
        Map<T, Integer> features = featureCountPerCategory.get(category);
        if (features == null) {
            return;
        }
        Integer count = features.get(feature);
        if (count == null) {
            return;
        }
        if (count == 1) {
            features.remove(feature);
            if (features.isEmpty()) {
                this.featureCountPerCategory.remove(category);
            }
        } else {
            features.put(feature, --count);
        }

        Integer totalCount = this.totalFeatureCount.get(feature);
        if (totalCount == null) {
            return;
        }
        if (totalCount == 1) {
            this.totalFeatureCount.remove(feature);
        } else {
            this.totalFeatureCount.put(feature, --totalCount);
        }
    }

    /**
     * Decrements the count of a given category. This is equal to telling the
     * classifier, that this category has occurred once less.
     *
     * @param category The category, which count to increase.
     */
    public void decrementCategory(K category) {
        Integer count = totalCategoryCount.get(category);
        if (count == null) {
            return;
        }
        if (count == 1) {
            totalCategoryCount.remove(category);
        } else {
            totalCategoryCount.put(category, --count);
        }
    }

    /**
     * Retrieves the number of occurrences of the given feature in the given
     * category.
     *
     * @param feature  The feature, which count to retrieve.
     * @param category The category, which the feature occurred in.
     * @return The number of occurrences of the feature in the category.
     */
    public int getFeatureCount(T feature, K category) {
        Map<T, Integer> features = featureCountPerCategory.get(category);
        if (features == null) return 0;
        return Optional.ofNullable(features.get(feature)).orElse(0);
    }

    /**
     * Retrieves the total number of occurrences of the given feature.
     *
     * @param feature The feature, which count to retrieve.
     * @return The total number of occurrences of the feature.
     */
    public int getFeatureCount(T feature) {
        return Optional.ofNullable(totalFeatureCount.get(feature)).orElse(0);
    }

    /**
     * Retrieves the number of occurrences of the given category.
     *
     * @param category The category, which count should be retrieved.
     * @return The number of occurrences.
     */
    public int getCategoryCount(K category) {
        return Optional.ofNullable(totalCategoryCount.get(category)).orElse(0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double featureProbability(T feature, K category) {
        // Total feature count
        final double featureCount = getFeatureCount(feature);
        if (featureCount == 0) {
            return 0;
        } else {
            return getFeatureCount(feature, category) / featureCount;
        }
    }

    /**
     * Retrieves the weighed average <code>P(feature|category)</code> with
     * overall weight of <code>1.0</code> and an assumed probability of
     * <code>0.5</code>. The probability defaults to the overall feature
     * probability.
     *
     * @param feature  The feature, which probability to calculate.
     * @param category The category.
     * @return The weighed average probability.
     * @see Classifier#featureProbability(Object, Object)
     * @see Classifier#featureWeighedAverage(Object,
     * Object, IFeatureProbability, double, double)
     */
    public double featureWeighedAverage(T feature, K category) {
        return this.featureWeighedAverage(feature, category, null, 1.0, 0.5);
    }

    /**
     * Retrieves the weighed average <code>P(feature|category)</code> with
     * overall weight of <code>1.0</code>, an assumed probability of
     * <code>0.5</code> and the given object to use for probability calculation.
     *
     * @param feature    The feature, which probability to calculate.
     * @param category   The category.
     * @param calculator The calculating object.
     * @return The weighed average probability.
     * @see Classifier#featureWeighedAverage(Object,
     * Object, IFeatureProbability, double, double)
     */
    public double featureWeighedAverage(T feature, K category, IFeatureProbability<T, K> calculator) {
        return this.featureWeighedAverage(feature, category, calculator, 1.0, 0.5);
    }

    /**
     * Retrieves the weighed average <code>P(feature|category)</code> with the
     * given weight and an assumed probability of <code>0.5</code> and the given
     * object to use for probability calculation.
     *
     * @param feature    The feature, which probability to calculate.
     * @param category   The category.
     * @param calculator The calculating object.
     * @param weight     The feature weight.
     * @return The weighed average probability.
     * @see Classifier#featureWeighedAverage(Object,
     * Object, IFeatureProbability, double, double)
     */
    public double featureWeighedAverage(T feature, K category, IFeatureProbability<T, K> calculator, double weight) {
        return this.featureWeighedAverage(feature, category, calculator, weight, 0.5);
    }

    /**
     * Retrieves the weighed average <code>P(feature|category)</code> with the
     * given weight, the given assumed probability and the given object to use
     * for probability calculation.
     *
     * @param feature            The feature, which probability to calculate.
     * @param category           The category.
     * @param calculator         The calculating object.
     * @param weight             The feature weight.
     * @param assumedProbability The assumed probability.
     * @return The weighed average probability.
     */
    public double featureWeighedAverage(T feature, K category, IFeatureProbability<T, K> calculator, double weight,
                                        double assumedProbability) {

        /*
         * use the given calculating object or the default method to calculate
         * the probability that the given feature occurred in the given
         * category.
         */
        final double basicProbability = (calculator == null) ? featureProbability(feature, category)
                : calculator.featureProbability(feature, category);

        int totals = Optional.ofNullable(totalFeatureCount.get(feature)).orElse(0);
        return (weight * assumedProbability + totals * basicProbability) / (weight + totals);
    }

    /**
     * Train the classifier by telling it that the given features resulted in
     * the given category.
     *
     * @param category The category the features belong to.
     * @param features The features that resulted in the given category.
     */
    public void learn(K category, Collection<T> features) {
        this.learn(new Classification<>(features, category));
    }

    /**
     * Train the classifier by telling it that the given features resulted in
     * the given category.
     *
     * @param classification The classification to learn.
     */
    public void learn(Classification<T, K> classification) {
        classification.getFeatures()
                .forEach(feature -> incrementFeature(feature, classification.getCategory()));

        this.incrementCategory(classification.getCategory());
        this.memoryQueue.offer(classification);

        if (this.memoryQueue.size() > this.memoryCapacity) {
            Classification<T, K> toForget = this.memoryQueue.remove();
            toForget.getFeatures()
                    .forEach(feature -> decrementFeature(feature, toForget.getCategory()));

            decrementCategory(toForget.getCategory());
        }
    }

    /**
     * The classify method. It will retrieve the most likely category for the
     * features given and depends on the concrete classifier implementation.
     *
     * @param features The features to classify.
     * @return The category most likely.
     */
    public abstract Classification<T, K> classify(Collection<T> features);
}
