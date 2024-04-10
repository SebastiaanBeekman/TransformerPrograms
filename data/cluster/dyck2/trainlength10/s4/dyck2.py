import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys[: i + 1]) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys[: i + 1]] for i, q in enumerate(queries)]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/dyck2/trainlength10/s4/dyck2_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 15}:
            return k_position == 17
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {17, 3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 19
        elif q_position in {18, 12}:
            return k_position == 15
        elif q_position in {13, 14}:
            return k_position == 18
        elif q_position in {16, 19}:
            return k_position == 16

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"("}:
            return k_token == "{"
        elif q_token in {")"}:
            return k_token == ")"
        elif q_token in {"<s>"}:
            return k_token == ""
        elif q_token in {"{"}:
            return k_token == "("
        elif q_token in {"}"}:
            return k_token == "}"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"("}:
            return k_token == "{"
        elif q_token in {")"}:
            return k_token == ")"
        elif q_token in {"<s>"}:
            return k_token == ""
        elif q_token in {"{"}:
            return k_token == "("
        elif q_token in {"}"}:
            return k_token == "}"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(token, position):
        if token in {"{", "("}:
            return position == 5
        elif token in {"}", ")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 11

    attn_0_3_pattern = select_closest(positions, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"{", "("}:
            return position == 19
        elif token in {"}", ")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 5

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 1, 6, 7, 8, 9, 14, 15}:
            return token == ""
        elif position in {2, 5}:
            return token == "<s>"
        elif position in {3, 4, 11, 12, 13, 16, 17, 18, 19}:
            return token == "("
        elif position in {10}:
            return token == "{"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
        if q_token in {"{", "("}:
            return k_token == ""
        elif q_token in {")"}:
            return k_token == "{"
        elif q_token in {"<s>"}:
            return k_token == ")"
        elif q_token in {"}"}:
            return k_token == "("

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_token, k_token):
        if q_token in {"{", "("}:
            return k_token == ""
        elif q_token in {")"}:
            return k_token == "("
        elif q_token in {"}", "<s>"}:
            return k_token == "{"

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output, attn_0_2_output):
        key = (attn_0_3_output, attn_0_2_output)
        if key in {
            ("(", "}"),
            (")", "("),
            (")", ")"),
            (")", "<s>"),
            (")", "}"),
            ("<s>", "}"),
            ("}", "("),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "}"),
        }:
            return 6
        elif key in {(")", "{"), ("<s>", ")"), ("}", "{")}:
            return 4
        elif key in {
            ("(", "{"),
            ("<s>", "{"),
            ("{", "("),
            ("{", ")"),
            ("{", "<s>"),
            ("{", "{"),
            ("{", "}"),
        }:
            return 7
        return 1

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_2_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(token, attn_0_0_output):
        key = (token, attn_0_0_output)
        if key in {
            ("(", "("),
            (")", "("),
            ("<s>", "("),
            ("<s>", "<s>"),
            ("{", "("),
            ("{", "<s>"),
        }:
            return 16
        elif key in {
            ("(", ")"),
            ("(", "<s>"),
            ("(", "}"),
            ("<s>", ")"),
            ("<s>", "}"),
            ("}", "("),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "}"),
        }:
            return 19
        elif key in {("(", "{"), (")", ")"), (")", "<s>"), (")", "{"), ("{", ")")}:
            return 15
        elif key in {("<s>", "{"), ("{", "{")}:
            return 7
        elif key in {("}", "{")}:
            return 18
        return 5

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(tokens, attn_0_0_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        if key in {
            ("(", "("),
            ("(", "<s>"),
            ("<s>", "("),
            ("{", "("),
            ("}", "{"),
            ("}", "}"),
        }:
            return 10
        elif key in {(")", "{"), ("<s>", "{"), ("{", "{")}:
            return 15
        elif key in {("}", "("), ("}", "<s>")}:
            return 8
        elif key in {("<s>", "<s>"), ("{", "<s>")}:
            return 19
        elif key in {("(", "{")}:
            return 1
        return 16

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_3_output, attn_0_0_output):
        key = (attn_0_3_output, attn_0_0_output)
        return 7

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_0_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_3_output):
        key = (num_attn_0_2_output, num_attn_0_3_output)
        return 12

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 19

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_3_output, num_attn_0_1_output):
        key = (num_attn_0_3_output, num_attn_0_1_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (3, 17),
            (3, 18),
            (3, 19),
        }:
            return 0
        return 12

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 16

    num_mlp_0_3_outputs = [num_mlp_0_3(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_0_output, position):
        if attn_0_0_output in {"("}:
            return position == 1
        elif attn_0_0_output in {")"}:
            return position == 2
        elif attn_0_0_output in {"<s>"}:
            return position == 8
        elif attn_0_0_output in {"{"}:
            return position == 7
        elif attn_0_0_output in {"}"}:
            return position == 5

    attn_1_0_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_3_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_0_output, mlp_0_2_output):
        if attn_0_0_output in {"{", "(", ")", "<s>"}:
            return mlp_0_2_output == 15
        elif attn_0_0_output in {"}"}:
            return mlp_0_2_output == 1

    attn_1_1_pattern = select_closest(mlp_0_2_outputs, attn_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, position):
        if token in {"{", "(", "<s>"}:
            return position == 1
        elif token in {")"}:
            return position == 2
        elif token in {"}"}:
            return position == 3

    attn_1_2_pattern = select_closest(positions, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_3_output, position):
        if attn_0_3_output in {"{", "(", ")", "<s>"}:
            return position == 1
        elif attn_0_3_output in {"}"}:
            return position == 4

    attn_1_3_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_0_output, token):
        if attn_0_0_output in {"("}:
            return token == "}"
        elif attn_0_0_output in {"}", ")"}:
            return token == ""
        elif attn_0_0_output in {"{", "<s>"}:
            return token == ")"

    num_attn_1_0_pattern = select(tokens, attn_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_3_output, token):
        if attn_0_3_output in {"{", "("}:
            return token == ""
        elif attn_0_3_output in {"<s>", ")"}:
            return token == "("
        elif attn_0_3_output in {"}"}:
            return token == "{"

    num_attn_1_1_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(num_mlp_0_3_output, mlp_0_1_output):
        if num_mlp_0_3_output in {0, 1, 2, 3, 4, 8, 9, 10, 12, 15, 16, 18}:
            return mlp_0_1_output == 15
        elif num_mlp_0_3_output in {13, 5}:
            return mlp_0_1_output == 5
        elif num_mlp_0_3_output in {6}:
            return mlp_0_1_output == 17
        elif num_mlp_0_3_output in {17, 19, 14, 7}:
            return mlp_0_1_output == 19
        elif num_mlp_0_3_output in {11}:
            return mlp_0_1_output == 11

    num_attn_1_2_pattern = select(
        mlp_0_1_outputs, num_mlp_0_3_outputs, num_predicate_1_2
    )
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(num_mlp_0_2_output, mlp_0_2_output):
        if num_mlp_0_2_output in {0, 1, 3, 4, 7, 9, 11, 12, 14, 16, 17, 18}:
            return mlp_0_2_output == 8
        elif num_mlp_0_2_output in {2, 5, 6, 10, 13, 15, 19}:
            return mlp_0_2_output == 15
        elif num_mlp_0_2_output in {8}:
            return mlp_0_2_output == 7

    num_attn_1_3_pattern = select(
        mlp_0_2_outputs, num_mlp_0_2_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, attn_1_1_output):
        key = (attn_1_2_output, attn_1_1_output)
        return 17

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_1_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_3_output, attn_1_3_output):
        key = (attn_0_3_output, attn_1_3_output)
        if key in {("<s>", "("), ("<s>", ")"), ("<s>", "<s>"), ("<s>", "}")}:
            return 8
        elif key in {("(", "{"), ("<s>", "{")}:
            return 13
        return 2

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_1_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_1_output, attn_0_0_output):
        key = (attn_1_1_output, attn_0_0_output)
        return 0

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_0_0_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_3_output, attn_1_0_output):
        key = (attn_1_3_output, attn_1_0_output)
        if key in {
            ("(", 6),
            ("(", 13),
            ("(", 19),
            (")", 2),
            (")", 3),
            (")", 5),
            (")", 6),
            (")", 9),
            (")", 13),
            (")", 14),
            (")", 16),
            (")", 17),
            (")", 18),
            (")", 19),
            ("<s>", 6),
            ("<s>", 13),
            ("<s>", 14),
            ("<s>", 16),
            ("<s>", 17),
            ("<s>", 18),
            ("<s>", 19),
            ("}", 0),
            ("}", 1),
            ("}", 2),
            ("}", 3),
            ("}", 4),
            ("}", 5),
            ("}", 6),
            ("}", 9),
            ("}", 10),
            ("}", 11),
            ("}", 13),
            ("}", 14),
            ("}", 15),
            ("}", 16),
            ("}", 17),
            ("}", 18),
            ("}", 19),
        }:
            return 9
        elif key in {
            ("(", 1),
            ("(", 3),
            ("(", 4),
            ("(", 7),
            ("(", 8),
            ("(", 9),
            ("(", 10),
            ("(", 11),
            ("(", 14),
            ("(", 16),
            ("(", 17),
            ("(", 18),
            ("<s>", 1),
            ("<s>", 4),
            ("<s>", 7),
            ("<s>", 8),
            ("{", 1),
            ("{", 3),
            ("{", 4),
            ("{", 7),
            ("{", 8),
            ("{", 11),
            ("{", 14),
            ("{", 17),
        }:
            return 10
        elif key in {("(", 12), (")", 12), ("<s>", 12), ("{", 12), ("}", 12)}:
            return 16
        return 17

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_0_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        if key in {0}:
            return 18
        return 7

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output, num_attn_0_2_output):
        key = (num_attn_1_2_output, num_attn_0_2_output)
        if key in {
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (2, 28),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 36),
            (2, 37),
            (2, 38),
            (2, 39),
            (3, 36),
            (3, 37),
            (3, 38),
            (3, 39),
        }:
            return 2
        elif key in {
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (3, 31),
            (3, 32),
            (3, 33),
            (3, 34),
            (3, 35),
        }:
            return 19
        return 7

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_1_output, num_attn_1_2_output):
        key = (num_attn_1_1_output, num_attn_1_2_output)
        return 5

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_3_output):
        key = num_attn_1_3_output
        return 17

    num_mlp_1_3_outputs = [num_mlp_1_3(k0) for k0 in num_attn_1_3_outputs]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, position):
        if token in {"}", "{", "("}:
            return position == 3
        elif token in {")"}:
            return position == 1
        elif token in {"<s>"}:
            return position == 15

    attn_2_0_pattern = select_closest(positions, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, position):
        if token in {"}", "{", "(", "<s>"}:
            return position == 1
        elif token in {")"}:
            return position == 3

    attn_2_1_pattern = select_closest(positions, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_0_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, position):
        if token in {"("}:
            return position == 15
        elif token in {"}", "{", ")"}:
            return position == 4
        elif token in {"<s>"}:
            return position == 1

    attn_2_2_pattern = select_closest(positions, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, position):
        if token in {"}", "(", ")"}:
            return position == 1
        elif token in {"<s>"}:
            return position == 15
        elif token in {"{"}:
            return position == 3

    attn_2_3_pattern = select_closest(positions, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_3_output, attn_1_1_output):
        if attn_1_3_output in {"("}:
            return attn_1_1_output == 2
        elif attn_1_3_output in {")"}:
            return attn_1_1_output == 7
        elif attn_1_3_output in {"<s>"}:
            return attn_1_1_output == 6
        elif attn_1_3_output in {"{"}:
            return attn_1_1_output == 13
        elif attn_1_3_output in {"}"}:
            return attn_1_1_output == 1

    num_attn_2_0_pattern = select(attn_1_1_outputs, attn_1_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, mlp_0_2_output):
        if attn_1_1_output in {0, 1, 7, 9, 10}:
            return mlp_0_2_output == 8
        elif attn_1_1_output in {2, 3, 4, 11, 12, 13, 15, 16, 17, 18, 19}:
            return mlp_0_2_output == 15
        elif attn_1_1_output in {5}:
            return mlp_0_2_output == 5
        elif attn_1_1_output in {6}:
            return mlp_0_2_output == 19
        elif attn_1_1_output in {8}:
            return mlp_0_2_output == 12
        elif attn_1_1_output in {14}:
            return mlp_0_2_output == 6

    num_attn_2_1_pattern = select(mlp_0_2_outputs, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"("}:
            return k_attn_0_3_output == "}"
        elif q_attn_0_3_output in {"}", "<s>", ")"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"{"}:
            return k_attn_0_3_output == ")"

    num_attn_2_2_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_1_output, mlp_0_1_output):
        if attn_1_1_output in {0}:
            return mlp_0_1_output == 6
        elif attn_1_1_output in {1, 15, 14, 7}:
            return mlp_0_1_output == 5
        elif attn_1_1_output in {2}:
            return mlp_0_1_output == 14
        elif attn_1_1_output in {3, 4, 6, 8, 18}:
            return mlp_0_1_output == 15
        elif attn_1_1_output in {5, 10, 11, 13, 16, 19}:
            return mlp_0_1_output == 19
        elif attn_1_1_output in {9}:
            return mlp_0_1_output == 9
        elif attn_1_1_output in {12}:
            return mlp_0_1_output == 12
        elif attn_1_1_output in {17}:
            return mlp_0_1_output == 17

    num_attn_2_3_pattern = select(mlp_0_1_outputs, attn_1_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_1_output, attn_2_2_output):
        key = (attn_2_1_output, attn_2_2_output)
        if key in {(6, ")"), (6, "<s>"), (6, "}")}:
            return 4
        elif key in {(2, ")")}:
            return 19
        return 2

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_3_output, attn_0_1_output):
        key = (attn_1_3_output, attn_0_1_output)
        if key in {
            ("(", "("),
            ("(", ")"),
            ("(", "<s>"),
            (")", "<s>"),
            (")", "}"),
            ("}", "("),
            ("}", ")"),
            ("}", "<s>"),
        }:
            return 19
        elif key in {("(", "}"), ("<s>", "}"), ("}", "}")}:
            return 4
        return 16

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_0_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_2_0_output, attn_2_3_output):
        key = (attn_2_0_output, attn_2_3_output)
        if key in {
            (")", "("),
            (")", ")"),
            (")", "<s>"),
            (")", "{"),
            (")", "}"),
            ("<s>", ")"),
            ("<s>", "}"),
            ("{", ")"),
            ("}", "("),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "{"),
            ("}", "}"),
        }:
            return 11
        return 4

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_2_0_outputs, attn_2_3_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_2_2_output, attn_2_1_output):
        key = (attn_2_2_output, attn_2_1_output)
        return 17

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_1_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_3_output, num_attn_0_0_output):
        key = (num_attn_1_3_output, num_attn_0_0_output)
        return 10

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_3_output, num_attn_1_2_output):
        key = (num_attn_2_3_output, num_attn_1_2_output)
        if key in {(0, 0)}:
            return 4
        return 12

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_2_output):
        key = num_attn_1_2_output
        return 14

    num_mlp_2_2_outputs = [num_mlp_2_2(k0) for k0 in num_attn_1_2_outputs]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_1_output):
        key = num_attn_1_1_output
        return 7

    num_mlp_2_3_outputs = [num_mlp_2_3(k0) for k0 in num_attn_1_1_outputs]
    num_mlp_2_3_output_scores = classifier_weights.loc[
        [("num_mlp_2_3_outputs", str(v)) for v in num_mlp_2_3_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                mlp_0_2_output_scores,
                mlp_0_3_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                num_mlp_0_2_output_scores,
                num_mlp_0_3_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                mlp_1_2_output_scores,
                mlp_1_3_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                num_mlp_1_2_output_scores,
                num_mlp_1_3_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                mlp_2_2_output_scores,
                mlp_2_3_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                num_mlp_2_2_output_scores,
                num_mlp_2_3_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(run(["<s>", "{", "}", ")", ")", "(", "}", "(", "{", ")"]))
