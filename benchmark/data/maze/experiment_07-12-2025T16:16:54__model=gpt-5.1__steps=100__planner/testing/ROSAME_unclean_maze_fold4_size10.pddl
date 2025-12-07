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
	:precondition (and (move-dir-up ?from ?to) (move-dir-up ?to ?from) (move-dir-down ?to ?from) (move-dir-left ?from ?to) (move-dir-right ?from ?to) (clear ?to) (at ?p ?from) (oriented-right ?p) (is-goal ?to))
	:effect (and (move-dir-down ?from ?to) (move-dir-left ?to ?from) (move-dir-right ?to ?from) (clear ?from) (at ?p ?to) (oriented-up ?p) (oriented-down ?p) (oriented-left ?p) (is-goal ?from) (not (move-dir-up ?to ?from))  (not (move-dir-left ?from ?to))  (not (move-dir-right ?from ?to))  (not (at ?p ?from))  (not (oriented-right ?p))  (not (is-goal ?to))))

(:action move-down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-up ?to ?from) (move-dir-left ?to ?from) (move-dir-right ?from ?to) (move-dir-right ?to ?from) (at ?p ?from) (at ?p ?to) (oriented-left ?p))
	:effect (and  (not (move-dir-up ?to ?from))  (not (move-dir-left ?to ?from))  (not (move-dir-right ?from ?to))  (not (move-dir-right ?to ?from))  (not (at ?p ?from))  (not (at ?p ?to))  (not (oriented-left ?p))))

(:action move-left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (clear ?to) (at ?p ?from) (oriented-up ?p) (is-goal ?to))
	:effect (and (move-dir-up ?from ?to) (move-dir-down ?from ?to) (move-dir-down ?to ?from) (move-dir-left ?from ?to) (move-dir-left ?to ?from) (move-dir-right ?from ?to) (move-dir-right ?to ?from) (clear ?from) (at ?p ?to) (oriented-down ?p) (oriented-left ?p) (oriented-right ?p) (is-goal ?from) (not (at ?p ?from))  (not (oriented-up ?p))))

(:action move-right
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (clear ?to) (at ?p ?from))
	:effect (and (move-dir-up ?from ?to) (move-dir-up ?to ?from) (move-dir-down ?from ?to) (move-dir-down ?to ?from) (move-dir-left ?from ?to) (move-dir-left ?to ?from) (move-dir-right ?from ?to) (move-dir-right ?to ?from) (clear ?from) (at ?p ?to) (oriented-up ?p) (oriented-down ?p) (oriented-left ?p) (oriented-right ?p) (is-goal ?from) (is-goal ?to) (not (clear ?to))  (not (at ?p ?from))))

)