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
	:precondition (and (clear ?from) (clear ?to))
	:effect (and (move-dir-up ?from ?to) (move-dir-up ?to ?from) (move-dir-down ?from ?to) (move-dir-down ?to ?from) (move-dir-left ?from ?to) (move-dir-left ?to ?from) (move-dir-right ?from ?to) (move-dir-right ?to ?from) (at ?p ?from) (at ?p ?to) (oriented-up ?p) (oriented-down ?p) (oriented-left ?p) (oriented-right ?p) (is-goal ?from) (is-goal ?to) (not (clear ?from))  (not (clear ?to))))

(:action move_down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-up ?to ?from) (move-dir-down ?from ?to) (clear ?from) (clear ?to) (oriented-up ?p))
	:effect (and (at ?p ?from) (oriented-right ?p) (not (clear ?from))  (not (oriented-up ?p))))

(:action move_left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-left ?from ?to) (move-dir-right ?to ?from) (clear ?to) (oriented-right ?p))
	:effect (and (clear ?from) (at ?p ?to) (not (clear ?to))))

(:action move_right
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-left ?to ?from) (move-dir-right ?from ?to) (clear ?from) (clear ?to) (oriented-right ?p))
	:effect (and (at ?p ?to) (is-goal ?to) (not (clear ?to))))

)