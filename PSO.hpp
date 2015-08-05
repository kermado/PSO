#ifndef PSO_HPP
#define PSO_HPP

#include <array>
#include <vector>
#include <initializer_list>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <algorithm>
#include <limits>
#include <cassert>

namespace
{
    /**
     * Computes a random number in an open interval of uniform distribution.
     *
     * @param lower Lower bound.
     * @param upper Upper bound.
     * @return Uniformly distributed random number in the interval [lower, upper].
     */
    float rand_in_range(float lower, float upper)
    {
        return lower + (std::rand() / static_cast <float>(RAND_MAX))  * (upper - lower);
    }
}

/**
 * Vector type providing the minimum required functionality for the Particle Swarm Optimisation solver.
 */
template<class T>
class Vector
{
public:
    /**
     * Default constructor.
     */
    Vector() = default;

    /**
     * Constructor.
     *
     * @param size Number of elements.
     * @param value Initial value for all elements.
     */
    Vector(std::size_t size, const T& value = T())
    : m_elements(size, value)
    {
        // Nothing to do.
    }

    Vector(std::initializer_list<T> values)
    : m_elements(values)
    {
        // Nothing to do.
    }

    /**
     * Destructor.
     */
    ~Vector() = default;

    /**
     * Assignment operator.
     *
     * @param other Other vector to copy from.
     * @return Reference to this vector.
     */
    Vector<T>& operator =(const Vector<T>& other)
    {
        if (this != &other)
        {
            m_elements.resize(other.size());
            for (std::size_t i = 0; i < size(); ++i)
            {
                m_elements[i] = other[i];
            }
        }

        return *this;
    }

    /**
     * Element access operator.
     *
     * Undefined behaviour when the provided index is not in the range [0, size).
     *
     * @param i Index for the element.
     * @return Element value at index `i`.
     */
    T operator [](std::size_t i) const
    {
        return m_elements[i];
    }

    /**
     * Element access operator.
     *
     * Undefined behaviour when the provided index is not in the range [0, size).
     *
     * @param i Index for the element.
     * @return Reference to the element at index `i`.
     */
    T& operator [](std::size_t i)
    {
        return m_elements[i];
    }

    /**
     * Addition assignment operator.
     *
     * Undefined behaviour when the sizes of the two vectors are not equal.
     *
     * @return Reference to this vector.
     */
    Vector<T>& operator +=(const Vector<T>& v)
    {
        for (std::size_t i = 0; i < size(); ++i)
        {
            m_elements[i] += v[i];
        }

        return *this;
    }

    /**
     * Addition operator.
     *
     * Undefined behaviour when the sizes of the two vectors are not equal.
     */
    Vector<T> operator +(const Vector<T>& v)
    {
        Vector<T> result(size());
        for (std::size_t i = 0; i < v.size(); ++i)
        {
            result[i] = m_elements[i] + v[i];
        }

        return result;
    }

    /**
     * Subtraction operator.
     *
     * Undefined behaviour when the sizes of the two vectors are not equal.
     */
    Vector<T> operator -(const Vector<T>& v)
    {
        Vector<T> result(size());
        for (std::size_t i = 0; i < v.size(); ++i)
        {
            result[i] = (m_elements[i] - v[i]);
        }

        return result;
    }

    /**
     * Scalar multiplication operator.
     *
     * @param s Scalar multiple.
     * @return Copy of this vector scaled by `s`.
     */
    Vector<T> operator *(float s)
    {
        Vector<T> result(size());
        for (std::size_t i = 0; i < size(); ++i)
        {
            result[i] = m_elements[i] * s;
        }

        return result;
    }

    /**
     * Returns the size of the vector.
     *
     * @return Number of elements in the vector.
     */
    std::size_t size() const
    {
        return m_elements.size();
    }

    /**
     * Returns the maximum value in the vector.
     *
     * Undefined behaviour when the vector is empty.
     *
     * @return Maximum element value.
     */
    T max() const
    {
        T max = m_elements[0];
        for (std::size_t i = 1; i < size(); ++i)
        {
            const T current = m_elements[i];
            if (current > max) max = current;
        }

        return max;
    }

    /**
     * Returns the minimum value in the vector.
     *
     * Undefined behaviour when the vector is empty.
     *
     * @return Maximum element value.
     */
    T min() const
    {
        T min = m_elements[0];
        for (std::size_t i = 1; i < size(); ++i)
        {
            const T current = m_elements[i];
            if (current < min) min = current;
        }

        return min;
    }

private:
    /**
     * Vector elements.
     */
    std::vector<T> m_elements;
};

/**
 * Represents a particle having position and velocity in an n-dimensional space.
 */
struct Particle
{
    /**
     * Remove default constructor.
     */
    Particle() = delete;

    /**
     * Constructor.
     *
     * The contents of the `position`, `velocity` and `best_position` vectors are left uninitialised.
     *
     * @param dimensions Number of dimensions for the particle's position and velocity vectors.
     */
    Particle(std::size_t dimensions)
    : position(dimensions)
    , velocity(dimensions)
    , best_position(dimensions)
    , best_objective(std::numeric_limits<float>::max())
    {
        // Nothing to do.
    }

    /**
     * Destructor.
     */
    ~Particle() = default;

    /**
     * Position.
     */
    Vector<float> position;

    /**
     * Velocity.
     */
    Vector<float> velocity;

    /**
     * Past position that achieved the particle's best achieved objective function value.
     */
    Vector<float> best_position;

    /**
     * Best achieved objective function for the particle.
     */
    float best_objective;
};

/**
 * Simplified Particle Swarm Optimisation solver with reasonable default parameter values.
 */
class Swarm
{
private:
    /**
     * Objective function type.
     */
    typedef std::function<float(const Vector<float>&)> ObjectiveFunction;

public:
    /**
     * Constructor.
     *
     * @param size Number of particles in the swarm.
     * @param dimensions Number of dimensions for the search space.
     */
    Swarm(std::size_t size, std::size_t dimensions)
    : m_dimensions(dimensions)
    , m_particles(size, Particle(dimensions))
    {
        std::srand(std::time(0)); // Seed random number generator with current time.
    }

    /**
     * Runs the basic Particle Swarm Optimisation algorithm with the objective function and initial variable values
     * provided.
     *
     * @param obj Objective function.
     * @param var Array of variables.
     * @param lower Lower bounds on the array of variable values.
     * @param upper Upper bounds on the array of variable values.
     * @param maxiter Maximum number of iterations.
     * @return Minimal objective function value obtained.
     */
    float solve(ObjectiveFunction obj, Vector<float>& var, Vector<float>& lower, Vector<float>& upper, unsigned int maxiter)
    {
        // Dimension consistency check.
        assert(var.size() == m_dimensions && lower.size() == m_dimensions && upper.size() == m_dimensions);

        // Sensible default parameter values for the social and personal particle influence weightings.
        const float personal = 0.5f;
        const float social = 0.5f;

        // Calculate a sensible upper bound on the initial particle velocities.
        const float v_max = (upper.max() - lower.min()) / maxiter;

        // Initialise particle positions and velocities within the respective bounds (uniform random distribution).
        for (unsigned int p = 0; p < m_particles.size(); ++p)
        {
            for (unsigned int n = 0; n < m_dimensions; ++n)
            {
                m_particles[p].position[n] = rand_in_range(lower[n], upper[n]);
                m_particles[p].velocity[n] = rand_in_range(0.0f, v_max);
            }
        }

        // Optimisation loop.
        float global_objective = std::numeric_limits<float>::max(); // Best objective function value achieved across all iterations.
        for (unsigned int k = 0; k < maxiter; ++k)
        {
            for (Particle& particle : m_particles)
            {
                // Compute random numbers used to update the particle's velocity.
                float r1 = rand_in_range(0.0f, 1.0f);
                float r2 = rand_in_range(0.0f, 1.0f);

                // Evaluate the objective function for the particle.
                const float current_objective = obj(particle.position);

                // Update the particle's current best values and the global best values as necessary.
                if (current_objective <= particle.best_objective)
                {
                    particle.best_objective = current_objective;
                    particle.best_position = particle.position;

                    if (current_objective <= global_objective)
                    {
                        global_objective = current_objective;
                        var = particle.position;
                    }
                }

                // Update the particle's velocity.
                particle.velocity += (particle.best_position - particle.position) * (personal * r1)
                                   + (var - particle.position) * (social * r2);

                // Update the particle's position for the next iteration.
                particle.position += particle.velocity;
            }
        }

        // Return the best objective function value obtained.
        return global_objective;
    }

private:
    /**
     * Number of dimensions for the search space.
     */
    std::size_t m_dimensions;

    /**
     * Particles in the search space.
     */
    std::vector<Particle> m_particles;
};

#endif