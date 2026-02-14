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
	:precondition (and (move-dir-up ?from ?to) (clear ?to) (at ?p ?from))
	:effect (and (clear ?from) (at ?p ?to) (oriented-up ?p) (not (clear ?to))  (not (at ?p ?from))))

(:action move_down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-up ?from ?to) (move-dir-up ?to ?from) (move-dir-down ?from ?to) (move-dir-down ?to ?from) (move-dir-left ?from ?to) (move-dir-left ?to ?from) (move-dir-right ?from ?to) (move-dir-right ?to ?from) (clear ?from) (clear ?to) (at ?p ?from) (at ?p ?to) (oriented-up ?p) (oriented-down ?p) (oriented-left ?p) (oriented-right ?p) (is-goal ?from) (is-goal ?to))
	:effect (and  (not (move-dir-up ?from ?to))  (not (move-dir-up ?to ?from))  (not (move-dir-down ?from ?to))  (not (move-dir-down ?to ?from))  (not (move-dir-left ?from ?to))  (not (move-dir-left ?to ?from))  (not (move-dir-right ?from ?to))  (not (move-dir-right ?to ?from))  (not (clear ?from))  (not (clear ?to))  (not (at ?p ?from))  (not (at ?p ?to))  (not (oriented-up ?p))  (not (oriented-down ?p))  (not (oriented-left ?p))  (not (oriented-right ?p))  (not (is-goal ?from))  (not (is-goal ?to))))

(:action move_left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (clear ?to) (at ?p ?from) (oriented-up ?p))
	:effect (and (clear ?from) (at ?p ?to) (oriented-left ?p) (not (clear ?to))  (not (at ?p ?from))  (not (oriented-up ?p))))

(:action move_right
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-left ?to ?from) (move-dir-right ?from ?to) (clear ?to) (at ?p ?from) (oriented-up ?p))
	:effect (and (clear ?from) (at ?p ?to) (oriented-right ?p) (not (clear ?to))  (not (at ?p ?from))  (not (oriented-up ?p))))

)