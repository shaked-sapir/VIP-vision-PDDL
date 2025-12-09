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

(:action move-up
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-right ?to ?from) (clear ?to))
	:effect (and (move-dir-up ?from ?to) (move-dir-up ?to ?from) (move-dir-down ?from ?to) (move-dir-down ?to ?from) (move-dir-left ?from ?to) (move-dir-left ?to ?from) (move-dir-right ?from ?to) (clear ?from) (at ?p ?from) (at ?p ?to) (oriented-up ?p) (oriented-down ?p) (oriented-left ?p) (oriented-right ?p) (is-goal ?from) (is-goal ?to) (not (move-dir-right ?to ?from))  (not (clear ?to))))

(:action move-down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-down ?from ?to) (clear ?from) (at ?p ?from) (oriented-down ?p))
	:effect (and (move-dir-up ?from ?to) (move-dir-up ?to ?from) (move-dir-down ?to ?from) (move-dir-left ?from ?to) (move-dir-left ?to ?from) (move-dir-right ?from ?to) (move-dir-right ?to ?from) (clear ?to) (at ?p ?to) (oriented-up ?p) (oriented-left ?p) (oriented-right ?p) (is-goal ?from) (is-goal ?to) (not (clear ?from))))

(:action move-left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-left ?from ?to) (move-dir-right ?to ?from) (clear ?to) (at ?p ?from) (oriented-down ?p) (is-goal ?to))
	:effect (and (move-dir-up ?from ?to) (move-dir-up ?to ?from) (move-dir-down ?from ?to) (move-dir-down ?to ?from) (move-dir-left ?to ?from) (move-dir-right ?from ?to) (clear ?from) (at ?p ?to) (oriented-up ?p) (oriented-left ?p) (oriented-right ?p) (is-goal ?from) (not (at ?p ?from))  (not (oriented-down ?p))  (not (is-goal ?to))))

(:action move-right
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-up ?from ?to) (move-dir-down ?to ?from) (move-dir-left ?from ?to) (move-dir-left ?to ?from) (move-dir-right ?from ?to) (move-dir-right ?to ?from) (clear ?from) (oriented-right ?p) (is-goal ?from))
	:effect (and (move-dir-up ?to ?from) (move-dir-down ?from ?to) (clear ?to) (at ?p ?from) (at ?p ?to) (oriented-up ?p) (oriented-down ?p) (oriented-left ?p) (is-goal ?to)))

)