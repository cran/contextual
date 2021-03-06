#' @export
Policy <- R6::R6Class(
  portable = FALSE,
  class = FALSE,
  public = list(
    action        = NULL,      # action results (list)
    theta         = NULL,      # policy parameters theta (list)
    theta_to_arms = NULL,      # theta to arms "helper" (list)
    is_oracle     = NULL,      # is policy an oracle? (logical)
    class_name    = "Policy",  # policy name - required (character)
    initialize = function() {
      # Is called before the Policy instance has been cloned.
      self$theta  <- list()    # initializes theta list
      self$action <- list()    # initializes action list
      is_oracle   <- FALSE     # very seldom TRUE
      invisible(self)
    },
    post_initialization = function() {
      # Is called after a Simulator has cloned the Policy instance [number_of_simulations] times.
      # Do sim level random generation here.
      invisible(self)
    },
    set_parameters = function(context_params) {
      # Parameter initialisation happens here.
    },
    get_action = function(t, context) {
      # Selects an arm based on paramters in self$theta and the current context,
      # the index of the chosen arm through action$choice.
      stop("Policy$get_action() has not been implemented.", call. = FALSE)
    },
    set_reward = function(t, context, action, reward) {
      # Updates parameters in theta based on current context and
      # the reward that was awarded by the bandit for the policy's action$choice.
      stop("Policy$set_reward() has not been implemented.", call. = FALSE)
    },
    initialize_theta = function(k) {
      # Called by a policy's agent during contextual's initialization phase.

      # The optional "helper variable" self$theta_to_arms
      # is parsed here. That is, when self$theta_to_arms exists, it is copied
      # self$k times, and each copy is made available through self$theta.
      if (!is.null(self$theta_to_arms)) {
        for (param_index in seq_along(self$theta_to_arms)) {
          self$theta[[ names(self$theta_to_arms)[param_index] ]] <-
            rep(list(self$theta_to_arms[[param_index]]),k)
        }
      }
      self$theta
    }
  )
)

#' Policy: Superclass
#'
#' Parent or superclass of all \code{\{contextual\}} \code{Policy} subclasses.
#'
#' On every \emph{t} = \{1, \ldots, T\}, a policy receives \code{d} dimensional feature vector or
#' \code{d x k} dimensional matrix
#' \code{context$X}*, the current number of \code{\link{Bandit}} arms in \code{context$k}, and the current
#' number of contextual features in \code{context$d}.
#'
#' To make sure a policy supports both contextual feature vectors and matrices in \code{context$X}, it is
#' suggested any contextual policy makes use of \pkg{contextual}'s \code{get_arm_context(context, arm)}
#' utility function to obtain the current context for a particular arm, and \code{get_full_context(context)}
#' where a policy makes direct use of a \code{d x k} context matrix.
#'
#' It has to compute which of the \code{k}
#' \code{\link{Bandit}} arms to pull by taking into account this contextual information plus the policy's
#' current parameter values stored in the named list \code{theta}. On selecting an arm, the policy then
#' returns its index as \code{action$choice}.
#'
#' ![](3bpolicy.jpeg "contextual diagram: get context")
#'
#' On pulling a \code{\link{Bandit}} arm the policy receives a \code{\link{Bandit}} reward through
#' \code{reward$reward}. In combination with the current \code{context$X}* and \code{action$choice},
#' this reward can then be used to update to the policy's parameters as stored in list \code{theta}.
#'
#' ![](3dpolicy.jpeg "contextual diagram: get context")
#'
#' &ast; Note: in context-free scenario's, \code{context$X} can be omitted.
#'
#' @name Policy
#' @aliases get_action set_reward set_parameters initialize_theta policy theta
#'
#' @section Usage:
#' \preformatted{
#'   policy <- Policy$new()
#' }
#'
#' @section Methods:
#'
#' \describe{
#'   \item{\code{new()}}{
#'     Generates and initializes a new \code{Policy} object.
#'   }
#'
#'   \item{\code{get_action(t, context)}}{
#'      arguments:
#'      \itemize{
#'          \item \code{t}: integer, time step \code{t}.
#'          \item \code{context}: list, containing the current \code{context$X} (d x k context matrix),
#'          \code{context$k} (number of arms) and \code{context$d} (number of context features)
#'      }
#'      computes which arm to play based on the current values in named list \code{theta}
#'      and the current \code{context}. Returns a named list containing
#'      \code{action$choice}, which holds the index of the arm to play.
#'   }
#'
#'   \item{\code{set_reward(t, context, action, reward)}}{
#'      arguments:
#'      \itemize{
#'          \item \code{t}: integer, time step \code{t}.
#'          \item \code{context}: list, containing the current \code{context$X} (d x k context matrix),
#'          \code{context$k} (number of arms) and \code{context$d} (number of context features)
#'          (as set by \code{bandit}).
#'          \item \code{action}:  list, containing \code{action$choice} (as set by \code{policy}).
#'          \item \code{reward}:  list, containing \code{reward$reward} and, if available,
#'          \code{reward$optimal} (as set by \code{bandit}).
#'      }
#'    utilizes the above arguments to update and return the set of parameters in list \code{theta}.
#'    }
#'
#'   \item{\code{post_initialization()}}{
#'         Post-initialization happens after cloning the Policy instance \code{number_of_simulations} times.
#'         Do sim level random generation here.
#'   }
#'
#'   \item{\code{set_parameters()}}{
#'    Helper function, called during a Policy's initialisation, assigns the values
#'    it finds in list \code{self$theta_to_arms} to each of the Policy's k arms.
#'    The parameters defined here can then be accessed by arm index in the following way:
#'    \code{theta[[index_of_arm]]$parameter_name}.
#'   }
#'
#'   }
#'
#' @seealso
#'
#' Core contextual classes: \code{\link{Bandit}}, \code{\link{Policy}}, \code{\link{Simulator}},
#' \code{\link{Agent}}, \code{\link{History}}, \code{\link{Plot}}
#'
#' Bandit subclass examples: \code{\link{BasicBernoulliBandit}}, \code{\link{ContextualLogitBandit}},
#' \code{\link{OfflineReplayEvaluatorBandit}}
#'
#' Policy subclass examples: \code{\link{EpsilonGreedyPolicy}}, \code{\link{ContextualLinTSPolicy}}
NULL
