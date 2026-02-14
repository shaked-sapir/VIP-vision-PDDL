(define (domain maze)
(:requirements :strips :typing)
(:types 	player location - object
)

(:predicates (move-dir-up ?v0 - location ?v1 - location)
	(move-dir-down ?v0 - location ?v1 - location)
	(move-dir-left ?v0 - location ?v1 - location)
	(move-dir-right ?v0 - location ?v1 - location)
	(clear ?v0 - location)
	(at ?v0 - player ?v1 - location)
	(oriented-up ?v0 - player)
	(oriented-down ?v0 - player)
	(oriented-left ?v0 - player)
	(oriented-right ?v0 - player)
	(is-goal ?v0 - location)
)

(:action move_up
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-up ?from ?to) (move-dir-down ?to ?from) (clear ?from) (clear ?to) (oriented-left ?p))
	:effect (and (oriented-down ?p) (not (move-dir-up ?from ?to))  (not (move-dir-down ?to ?from))  (not (clear ?from))  (not (oriented-left ?p))))

(:action move_down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-up ?to ?from) (move-dir-down ?from ?to) (clear ?from) (clear ?to) (oriented-left ?p) (is-goal ?to))
	:effect (and (at ?p ?to) (oriented-down ?p) (not (clear ?to))  (not (oriented-left ?p))  (not (is-goal ?to))))

(:action move_left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and )
	:effect (and (oriented-left ?p) (is-goal ?to)))

(:action move_right
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (clear ?to) (oriented-down ?p))
	:effect (and (move-dir-left ?to ?from) (move-dir-right ?from ?to) (clear ?from) (oriented-left ?p) (oriented-right ?p) (not (clear ?to))  (not (oriented-down ?p))))

)