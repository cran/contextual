YahooEpsilonGreedyPolicy          <- R6::R6Class(
  portable = FALSE,
  class = FALSE,
  inherit = Policy,
  public = list(
    epsilon = NULL,
    class_name = "YahooEpsilonGreedyPolicy",
    initialize = function(epsilon = 0.1) {
      super$initialize()
      self$epsilon                <- epsilon
    },
    set_parameters = function(context_params) {
      self$theta_to_arms          <- list('n' = 0, 'mean' = 0)
    },
    get_action = function(t, context) {
      if (runif(1) > self$epsilon) {
        max_index                 <- context$arms[which_max_list(self$theta$mean[context$arms])]
        self$action$choice        <- max_index
      } else {
        self$action$choice        <- sample(context$arms, 1)
      }
      self$action
    },
    set_reward = function(t, context, action, reward) {

      arm                         <- action$choice
      reward                      <- reward$reward

      self$theta$n[[arm]]         <- self$theta$n[[arm]] + 1
      self$theta$mean[[arm]]      <- self$theta$mean[[arm]] + (reward - self$theta$mean[[arm]]) / self$theta$n[[arm]]

      self$theta
    }
  )
)
