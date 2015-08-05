#include "PSO.hpp"

#include <cmath>
#include <iostream>

/**
 * Runs some common benchmark optimisation problems. All functions have their global minima at xi = 0 (for all i).
 */
int main()
{
    // This first problem is the minimisation of the continuous function y = x^2.
    // There are nonlinear gradient descent methods that will converge to a solution much quicker than PSO, however it
    // is provided to illustrate solving a minimal single variable problem.
    {
        std::cout << "-----" << std::endl;

        const unsigned int particles = 100;
        Vector<float> variables(1);
        Vector<float> lower_bounds{-100.0f};
        Vector<float> upper_bounds{100.0f};

        auto x_squared = [](Vector<float> var) {
            return var[0] * var[0];
        };

        Swarm solver(particles, 1);
        const float result = solver.solve(x_squared, variables, lower_bounds, upper_bounds, 100);

        std::cout << "Optimisation for the function: y = x0^2" << std::endl;
        std::cout << "Minimum objective function: " << result << std::endl;
        std::cout << "Achieved with variable values: [" << variables[0] << "]" << std::endl;

        std::cout << "-----" << std::endl;
    }

    // Ackley's function is a multimodal problem over 2 dimensions that has many local minima but an easily
    // distinguishable global minimum.
    // https://en.wikipedia.org/wiki/File:Ackley%27s_function.pdf
    {
        std::cout << "-----" << std::endl;

        const unsigned int particles = 100;
        Vector<float> variables(2);
        Vector<float> lower_bounds{-5.0f, -5.0f};
        Vector<float> upper_bounds{5.0f, 5.0f};

        auto ackley = [](Vector<float> var) {
            return -20.0f * std::exp(-0.2f * std::sqrt(0.5f * (var[0] * var[0] + var[1] * var[1])))
                   - std::exp(0.5f *(std::cos(2.0f * M_PI * var[0]) + std::cos(2.0f * M_PI * var[1])))
                   + std::exp(1.0) + 20.0f;
        };

        Swarm solver(particles, 2);
        const float result = solver.solve(ackley, variables, lower_bounds, upper_bounds, 100);

        std::cout << "Optimisation for Ackley's function" << std::endl;
        std::cout << "Minimum objective function: " << result << std::endl;
        std::cout << "Achieved with variable values: [" << variables[0] << ", " << variables[1] << "]" << std::endl;

        std::cout << "-----" << std::endl;
    }

    // The Rastrigin function is a multimodal problem that has many local minima and a weak global minimum. The problem
    // has been generalised to n-dimensions.
    // https://en.wikipedia.org/wiki/Rastrigin_function#/media/File:Rastrigin_function.png
    {
        std::cout << "-----" << std::endl;

        const unsigned int particles = 100;
        const unsigned int dimensions = 2;
        Vector<float> variables(dimensions);
        Vector<float> lower_bounds{-5.12f, -5.12f};
        Vector<float> upper_bounds{5.12f, 5.12f};

        auto rastrigin = [dimensions](Vector<float> var) {
            float sum = 10.0f * dimensions;
            for (std::size_t i = 0; i < dimensions; ++i)
            {
                sum += (var[i] * var[i]) - 10.0f * std::cos(2.0f * M_PI * var[i]);
            }
            return sum;
        };

        Swarm solver(particles, dimensions);
        const float result = solver.solve(rastrigin, variables, lower_bounds, upper_bounds, 100);

        std::cout << "Optimisation for the " << dimensions << "-dimensional Rastrigin function" << std::endl;
        std::cout << "Minimum objective function: " << result << std::endl;
        std::cout << "Achieved with variable values: [";
        for (int i = 0; i < dimensions; ++i)
        {
            std::cout << variables[i];
            if (i < dimensions - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "-----" << std::endl;
    }

    return 0;
}