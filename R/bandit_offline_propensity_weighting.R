#' @export
#' @import Formula
OfflinePropensityWeightingBandit <- R6::R6Class(
  inherit = OfflineBootstrappedReplayBandit,
  class = FALSE,
  private = list(
    p = NULL,
    p_hat = NULL,
    n = NULL
  ),
  public = list(
    class_name = "OfflinePropensityWeightingBandit",
    threshold = NULL,
    drop_value = NULL,
    stabilized = NULL,
    initialize   = function(formula,
                            data, k = NULL, d = NULL,
                            unique = NULL, shared = NULL,
                            randomize = TRUE, replacement = FALSE,
                            jitter = FALSE, arm_multiply = FALSE, threshold = 0,
                            stabilized = TRUE, drop_unequal_arm = TRUE) {
      self$threshold  <- threshold
      self$stabilized <- stabilized
      if(isTRUE(drop_unequal_arm)) {
        self$drop_value = NULL
      } else {
        self$drop_value = 0
      }
      super$initialize(formula,
                       data, k, d,
                       unique, shared,
                       randomize, replacement,
                       jitter, arm_multiply)
    },
    post_initialization = function() {
      super$post_initialization()
      private$p <-  Formula::model.part(private$formula, data = private$S, lhs = 0, rhs = 3, drop = TRUE)
      if (length(private$p) == 0 || is.null(private$p)) {
        marginal_prob <- table(private$z)/length(private$z)
        private$p     <- marginal_prob[private$z]
      }
      private$n       <- 0
      private$p_hat   <- 0
    },
    get_reward = function(index, context, action) {
      if (private$z[[index]] == action$choice) {

        p <- max(private$p[index], self$threshold) # when threshold 0 (default)
                                                   # p = private$p[index]

        w <- 1 / p

        if (self$stabilized) {

          inc(private$n)     <- 1
          inc(private$p_hat) <- (w - private$p_hat) / private$n
          prop_reward        <- as.double((private$y[index]*w)/private$p_hat)

        } else {
          prop_reward        <- as.double(private$y[index]*w)
        }

        list(
          reward         = prop_reward,
          optimal_reward = ifelse(private$or, as.double(private$S$optimal_reward[[index]]), NA),
          optimal_arm    = ifelse(private$oa, as.double(private$S$optimal_arm[[index]]), NA)
        )
      } else {
        if(is.null(self$drop_value)) {
          return(NULL)
        } else {
          list(
            reward         = 0,
            optimal_reward = ifelse(private$or, as.double(private$S$optimal_reward[[index]]), NA),
            optimal_arm    = ifelse(private$oa, as.double(private$S$optimal_arm[[index]]), NA)
          )
        }
      }
    }
  )
)

#' Bandit: Offline Propensity Weighted Replay
#'
#' Policy for the evaluation of policies with offline data through replay with propensity weighting.
#'
#' @name OfflinePropensityWeightingBandit
#'
#'
#' @section Usage:
#' \preformatted{
#'   bandit <- OfflinePropensityWeightingBandit(formula,
#'                                              data, k = NULL, d = NULL,
#'                                              unique = NULL, shared = NULL,
#'                                              randomize = TRUE, replacement = TRUE,
#'                                              jitter = TRUE, arm_multiply = TRUE)
#' }
#'
#' @section Arguments:
#'
#' \describe{
#'   \item{\code{formula}}{
#'     formula (required). Format: \code{y.context ~ z.choice | x1.context + x2.xontext + ... | p.propensity }
#'     When leaving out p.propensity, Doubly Robust Bandit uses marginal prob per arm for propensities:
#      table(private$z)/length(private$z).
#'     By default,  adds an intercept to the context model. Exclude the intercept, by adding "0" or "-1" to
#'     the list of contextual features, as in:
#'     \code{y.context ~ z.choice | x1.context + x2.xontext -1 | p.propensity }
#'   }
#'   \item{\code{data}}{
#'     data.table or data.frame; offline data source (required)
#'   }
#'   \item{\code{k}}{
#'     integer; number of arms (optional). Optionally used to reformat the formula defined x.context vector
#'     as a \code{k x d} matrix. When making use of such matrix formatted contexts, you need to define custom
#'     intercept(s) when and where needed in data.table or data.frame.
#'   }
#'   \item{\code{d}}{
#'     integer; number of contextual features (optional) Optionally used to reformat the formula defined
#'     x.context vector as a \code{k x d} matrix. When making use of such matrix formatted contexts, you need
#'     to define custom intercept(s) when and where needed in data.table or data.frame.
#'   }
#'   \item{\code{randomize}}{
#'     logical; randomize rows of data stream per simulation (optional, default: TRUE)
#'   }
#'   \item{\code{replacement}}{
#'     logical; sample with replacement (optional, default: TRUE)
#'   }
#'   \item{\code{jitter}}{
#'     logical; add jitter to contextual features (optional, default: TRUE)
#'   }
#'   \item{\code{arm_multiply}}{
#'     logical; multiply the horizon by the number of arms (optional, default: TRUE)
#'   }
#'  \item{\code{threshold}}{
#'     float (0,1); Lower threshold or Tau on propensity score values. Smaller Tau makes for less biased
#'     estimates with more variance, and vice versa. For more information, see paper by Strehl at all (2010).
#'     Values between 0.01 and 0.05 are known to work well.
#'   }
#'  \item{\code{drop_value}}{
#'     logical; Whether to drop a sample when the chosen arm does not equal the sampled arm. When TRUE,
#'     the sample is dropped by setting the reward to null. When FALSE, the reward will be zero.
#'   }
#'  \item{\code{stabilized}}{
#'     logical; Whether to stabilize propensity weights.
#'     One common issue with inverse propensity weighting g is that samples with
#'     a propensity score very close to 0 will end up with an extremely large propensity weight,
#'     potentially making the weighted estimator highly unstable.
#'     A common alternative to the conventional weights are stabilized weights,
#'     which use the marginal probability of treatment instead of 1 in the weight numerator.
#'   }
#'   \item{\code{unique}}{
#'     integer vector; index of disjoint features (optional)
#'   }
#'   \item{\code{shared}}{
#'     integer vector; index of shared features (optional)
#'   }
#'
#' }
#'
#' @section Methods:
#'
#' \describe{
#'
#'   \item{\code{new(formula, data, k = NULL, d = NULL, unique = NULL, shared = NULL, randomize = TRUE,
#'                   replacement = TRUE, jitter = TRUE, arm_multiply = TRUE)}}{
#'                   generates and instantializes a new \code{OfflinePropensityWeightingBandit} instance. }
#'
#'   \item{\code{get_context(t)}}{
#'      argument:
#'      \itemize{
#'          \item \code{t}: integer, time step \code{t}.
#'      }
#'      returns a named \code{list}
#'      containing the current \code{d x k} dimensional matrix \code{context$X},
#'      the number of arms \code{context$k} and the number of features \code{context$d}.
#'  }
#'
#'   \item{\code{get_reward(t, context, action)}}{
#'      arguments:
#'      \itemize{
#'          \item \code{t}: integer, time step \code{t}.
#'          \item \code{context}: list, containing the current \code{context$X} (d x k context matrix),
#'          \code{context$k} (number of arms) and \code{context$d} (number of context features)
#'          (as set by \code{bandit}).
#'          \item \code{action}:  list, containing \code{action$choice} (as set by \code{policy}).
#'      }
#'      returns a named \code{list} containing \code{reward$reward} and, where computable,
#'         \code{reward$optimal} (used by "oracle" policies and to calculate regret).
#'  }
#'
#'   \item{\code{post_initialization()}}{
#'      Randomize offline data by shuffling the offline data.table before the start of each
#'      individual simulation when self$randomize is TRUE (default)
#'   }
#' }
#'
#' @references
#'
#' Agarwal, Alekh, et al. "Taming the monster: A fast and simple algorithm for contextual bandits."
#' International Conference on Machine Learning. 2014.
#'
#' Strehl, Alex, et al. "Learning from logged implicit exploration data." Advances in Neural Information
#' Processing Systems. 2010.
#'
#' @seealso
#'
#' Core contextual classes: \code{\link{Bandit}}, \code{\link{Policy}}, \code{\link{Simulator}},
#' \code{\link{Agent}}, \code{\link{History}}, \code{\link{Plot}}
#'
#' Bandit subclass examples: \code{\link{BasicBernoulliBandit}}, \code{\link{ContextualLogitBandit}},
#' \code{\link{OfflinePropensityWeightingBandit}}
#'
#' Policy subclass examples: \code{\link{EpsilonGreedyPolicy}}, \code{\link{ContextualLinTSPolicy}}
#'
#' @examples
#' \dontrun{
#'
#' library(contextual)
#' ibrary(data.table)
#'
#' # Import myocardial infection dataset
#'
#' url  <- "http://d1ie9wlkzugsxr.cloudfront.net/data_propensity/myocardial_propensity.csv"
#' data            <- fread(url)
#'
#' simulations     <- 3000
#' horizon         <- nrow(data)
#'
#' # arms always start at 1
#' data$trt        <- data$trt + 1
#'
#' # turn death into alive, making it a reward
#' data$alive      <- abs(data$death - 1)
#'
#' # calculate propensity weights
#'
#' m      <- glm(I(trt-1) ~ age + risk + severity, data=data, family=binomial(link="logit"))
#' data$p <-predict(m, type = "response")
#'
#' # run bandit - if you leave out p, Propensity Bandit uses marginal prob per arm for propensities:
#' # table(private$z)/length(private$z)
#'
#' f          <- alive ~ trt | age + risk + severity | p
#'
#' bandit     <- OfflinePropensityWeightingBandit$new(formula = f, data = data)
#'
#' # Define agents.
#' agents      <- list(Agent$new(LinUCBDisjointOptimizedPolicy$new(0.2), bandit, "LinUCB"))
#'
#' # Initialize the simulation.
#'
#' simulation  <- Simulator$new(agents = agents, simulations = simulations, horizon = horizon)
#'
#' # Run the simulation.
#' sim  <- simulation$run()
#'
#' # plot the results
#' plot(sim, type = "cumulative", regret = FALSE, rate = TRUE, legend_position = "bottomright")
#'
#' }
NULL
