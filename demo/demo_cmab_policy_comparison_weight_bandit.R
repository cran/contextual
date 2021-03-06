library(contextual)

horizon     <- 1000L
simulations <- 100L

weights     <- matrix(c(0.8, 0.1, 0.1,
                        0.1, 0.8, 0.1,
                        0.1, 0.1, 0.8), nrow = 3, ncol = 3, byrow = TRUE)

bandit      <- ContextualBinaryBandit$new(weights = weights)
agents      <- list(Agent$new(ContextualTSProbitPolicy$new(draws = 100), bandit, "TSProbit"),
                    Agent$new(ContextualEpsilonGreedyPolicy$new(0.1), bandit, "cEGreedy"),
                    Agent$new(ContextualLogitBTSPolicy$new(draws = 100), bandit, "LogitBTS"),
                    Agent$new(LinUCBDisjointPolicy$new(0.6), bandit, "LinUCB"))

simulation  <- Simulator$new(agents, horizon, simulations)
history     <- simulation$run()

plot(history, type = "cumulative", no_par = TRUE, legend_border = FALSE, legend_position = "topleft")
