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
        "output/length/rasp/dyck2/trainlength10/s1/dyck2_weights.csv",
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
    def predicate_0_0(token, position):
        if token in {"(", "{"}:
            return position == 1
        elif token in {")"}:
            return position == 5
        elif token in {"<s>"}:
            return position == 9
        elif token in {"}"}:
            return position == 4

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 14
        elif q_position in {8, 1, 2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 7
        elif q_position in {4}:
            return k_position == 1
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 3
        elif q_position in {9}:
            return k_position == 18
        elif q_position in {10, 15}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 15
        elif q_position in {12, 14}:
            return k_position == 19
        elif q_position in {13}:
            return k_position == 17
        elif q_position in {16, 18}:
            return k_position == 11
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {19}:
            return k_position == 13

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
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
        elif q_token in {"<s>", "{"}:
            return k_token == "("
        elif q_token in {"}"}:
            return k_token == "}"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 1, 16}:
            return k_position == 18
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
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
        elif q_position in {10, 13}:
            return k_position == 16
        elif q_position in {11}:
            return k_position == 15
        elif q_position in {19, 12}:
            return k_position == 13
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {15}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 11
        elif q_position in {18}:
            return k_position == 14

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 2, 6, 7, 8, 12, 14, 15, 16, 17, 18, 19}:
            return token == ""
        elif position in {1, 5}:
            return token == "<s>"
        elif position in {3, 4}:
            return token == ")"
        elif position in {9}:
            return token == "<pad>"
        elif position in {10, 11, 13}:
            return token == "{"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"("}:
            return k_token == "<pad>"
        elif q_token in {")"}:
            return k_token == "("
        elif q_token in {"}", "<s>"}:
            return k_token == "{"
        elif q_token in {"{"}:
            return k_token == ""

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
        if q_token in {"("}:
            return k_token == ""
        elif q_token in {")"}:
            return k_token == ")"
        elif q_token in {"}", "<s>", "{"}:
            return k_token == "}"

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 1, 2, 5, 6, 8, 10, 13, 18, 19}:
            return token == ")"
        elif position in {3, 4, 11, 12, 14, 15}:
            return token == "}"
        elif position in {16, 9, 17, 7}:
            return token == ""

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, attn_0_1_output):
        key = (attn_0_2_output, attn_0_1_output)
        if key in {
            ("(", "("),
            ("(", ")"),
            ("(", "<s>"),
            ("(", "}"),
            (")", "("),
            ("}", "{"),
        }:
            return 7
        elif key in {("{", "("), ("{", ")"), ("{", "<s>"), ("{", "{"), ("{", "}")}:
            return 12
        elif key in {(")", "{")}:
            return 10
        elif key in {("(", "{")}:
            return 1
        return 5

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, token):
        key = (attn_0_3_output, token)
        if key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            ("<s>", "("),
            ("<s>", "<s>"),
            ("<s>", "{"),
            ("{", "("),
            ("{", "{"),
        }:
            return 16
        elif key in {(")", ")"), (")", "}"), ("}", ")"), ("}", "}")}:
            return 8
        elif key in {
            (")", "("),
            (")", "<s>"),
            (")", "{"),
            ("}", "("),
            ("}", "<s>"),
            ("}", "{"),
        }:
            return 19
        elif key in {("(", "}"), ("<s>", "}")}:
            return 2
        elif key in {("(", ")"), ("{", "}")}:
            return 9
        return 5

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, tokens)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        if key in {("(", ")"), ("{", ")")}:
            return 11
        elif key in {
            ("(", "("),
            ("(", "}"),
            (")", "("),
            ("<s>", "("),
            ("{", "("),
            ("{", "}"),
            ("}", "("),
        }:
            return 4
        elif key in {("(", "{"), ("<s>", "{"), ("{", "{"), ("}", "{")}:
            return 17
        elif key in {("(", "<s>"), ("{", "<s>")}:
            return 16
        return 2

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_1_output, attn_0_3_output):
        key = (attn_0_1_output, attn_0_3_output)
        if key in {
            ("(", "("),
            ("(", "{"),
            (")", "("),
            ("<s>", "("),
            ("{", "("),
            ("}", "("),
        }:
            return 19
        elif key in {("(", "<s>"), (")", "<s>"), ("<s>", "<s>"), ("{", "<s>")}:
            return 1
        return 14

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_3_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_3_output):
        key = num_attn_0_3_output
        return 12

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output, num_attn_0_1_output):
        key = (num_attn_0_2_output, num_attn_0_1_output)
        if key in {
            (1, 0),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (3, 2),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 7),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 9),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (12, 8),
            (12, 9),
            (12, 10),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (13, 7),
            (13, 8),
            (13, 9),
            (13, 10),
            (13, 11),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (14, 7),
            (14, 8),
            (14, 9),
            (14, 10),
            (14, 11),
            (14, 12),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (15, 8),
            (15, 9),
            (15, 10),
            (15, 11),
            (15, 12),
            (15, 13),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 8),
            (16, 9),
            (16, 10),
            (16, 11),
            (16, 12),
            (16, 13),
            (16, 14),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (17, 8),
            (17, 9),
            (17, 10),
            (17, 11),
            (17, 12),
            (17, 13),
            (17, 14),
            (17, 15),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 7),
            (18, 8),
            (18, 9),
            (18, 10),
            (18, 11),
            (18, 12),
            (18, 13),
            (18, 14),
            (18, 15),
            (18, 16),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
            (19, 10),
            (19, 11),
            (19, 12),
            (19, 13),
            (19, 14),
            (19, 15),
            (19, 16),
            (19, 17),
        }:
            return 8
        elif key in {(8, 7), (9, 8), (10, 9), (11, 10)}:
            return 19
        return 16

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_3_output, num_attn_0_0_output):
        key = (num_attn_0_3_output, num_attn_0_0_output)
        return 7

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 9

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_1_output, position):
        if attn_0_1_output in {"(", "<s>", "{"}:
            return position == 1
        elif attn_0_1_output in {"}", ")"}:
            return position == 3

    attn_1_0_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 15
        elif q_position in {1, 12, 13, 15, 19}:
            return k_position == 10
        elif q_position in {10, 2, 3}:
            return k_position == 2
        elif q_position in {4, 5}:
            return k_position == 3
        elif q_position in {9, 6, 7}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {11, 14, 16, 17, 18}:
            return k_position == 1

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_1_output, position):
        if mlp_0_1_output in {0, 2, 3, 4, 5, 6, 7, 10}:
            return position == 1
        elif mlp_0_1_output in {16, 1, 19, 9}:
            return position == 3
        elif mlp_0_1_output in {8}:
            return position == 4
        elif mlp_0_1_output in {18, 11, 14}:
            return position == 15
        elif mlp_0_1_output in {12}:
            return position == 10
        elif mlp_0_1_output in {17, 13, 15}:
            return position == 17

    attn_1_2_pattern = select_closest(positions, mlp_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 1, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return k_position == 1
        elif q_position in {10, 2, 3, 7}:
            return k_position == 2
        elif q_position in {9, 4, 5, 6}:
            return k_position == 3
        elif q_position in {8}:
            return k_position == 5

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_1_output, mlp_0_2_output):
        if attn_0_1_output in {"}", ")", "(", "<s>", "{"}:
            return mlp_0_2_output == 2

    num_attn_1_0_pattern = select(mlp_0_2_outputs, attn_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(token, attn_0_0_output):
        if token in {"("}:
            return attn_0_0_output == ""
        elif token in {")"}:
            return attn_0_0_output == "{"
        elif token in {"}", "<s>"}:
            return attn_0_0_output == "("
        elif token in {"{"}:
            return attn_0_0_output == "}"

    num_attn_1_1_pattern = select(attn_0_0_outputs, tokens, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_1_output, mlp_0_1_output):
        if attn_0_1_output in {"}", "(", "<s>", "{"}:
            return mlp_0_1_output == 5
        elif attn_0_1_output in {")"}:
            return mlp_0_1_output == 2

    num_attn_1_2_pattern = select(mlp_0_1_outputs, attn_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_mlp_0_1_output, k_mlp_0_1_output):
        if q_mlp_0_1_output in {0, 8, 13}:
            return k_mlp_0_1_output == 13
        elif q_mlp_0_1_output in {1, 3, 4, 5, 16, 19}:
            return k_mlp_0_1_output == 9
        elif q_mlp_0_1_output in {2}:
            return k_mlp_0_1_output == 16
        elif q_mlp_0_1_output in {6, 12, 14, 15, 17, 18}:
            return k_mlp_0_1_output == 8
        elif q_mlp_0_1_output in {7}:
            return k_mlp_0_1_output == 12
        elif q_mlp_0_1_output in {9}:
            return k_mlp_0_1_output == 19
        elif q_mlp_0_1_output in {10}:
            return k_mlp_0_1_output == 7
        elif q_mlp_0_1_output in {11}:
            return k_mlp_0_1_output == 14

    num_attn_1_3_pattern = select(mlp_0_1_outputs, mlp_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(mlp_0_2_output, attn_1_3_output):
        key = (mlp_0_2_output, attn_1_3_output)
        return 7

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(mlp_0_2_outputs, attn_1_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_1_output, attn_1_3_output):
        key = (attn_1_1_output, attn_1_3_output)
        if key in {
            (0, 9),
            (0, 19),
            (1, 9),
            (1, 19),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 17),
            (2, 18),
            (2, 19),
            (3, 9),
            (3, 19),
            (4, 9),
            (4, 19),
            (6, 2),
            (6, 9),
            (6, 19),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 17),
            (8, 18),
            (8, 19),
            (11, 2),
            (11, 8),
            (11, 9),
            (11, 14),
            (11, 19),
            (13, 9),
            (14, 1),
            (14, 2),
            (14, 8),
            (14, 9),
            (14, 14),
            (14, 19),
            (18, 2),
            (18, 9),
            (18, 19),
        }:
            return 17
        elif key in {
            (0, 16),
            (2, 16),
            (3, 16),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 10),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (4, 16),
            (4, 17),
            (4, 18),
            (5, 4),
            (5, 16),
            (6, 4),
            (6, 16),
            (8, 16),
            (10, 4),
            (10, 16),
            (14, 16),
            (15, 4),
            (15, 16),
            (16, 4),
            (16, 16),
            (18, 16),
        }:
            return 3
        return 1

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_0_output, attn_1_1_output):
        key = (attn_1_0_output, attn_1_1_output)
        if key in {
            (1, 10),
            (2, 2),
            (2, 8),
            (2, 10),
            (2, 13),
            (5, 8),
            (5, 10),
            (8, 2),
            (8, 8),
            (8, 10),
            (8, 13),
            (9, 8),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 4),
            (10, 5),
            (10, 8),
            (10, 10),
            (10, 11),
            (10, 13),
            (10, 14),
            (11, 10),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 4),
            (13, 5),
            (13, 8),
            (13, 10),
            (13, 11),
            (13, 13),
            (13, 14),
            (17, 10),
            (19, 8),
        }:
            return 15
        elif key in {
            (0, 8),
            (0, 10),
            (0, 13),
            (1, 8),
            (1, 13),
            (3, 8),
            (3, 10),
            (3, 13),
            (4, 8),
            (4, 13),
            (6, 8),
            (6, 10),
            (6, 13),
            (7, 8),
            (10, 3),
            (11, 8),
            (12, 8),
            (13, 3),
            (13, 6),
            (13, 7),
            (13, 17),
            (13, 18),
            (14, 8),
            (14, 13),
            (15, 8),
            (16, 8),
            (16, 13),
            (17, 8),
            (17, 13),
            (18, 8),
            (18, 13),
            (19, 13),
        }:
            return 11
        elif key in {(9, 10), (14, 10), (19, 10)}:
            return 14
        elif key in {(4, 10), (10, 6)}:
            return 2
        return 7

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_1_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_3_output, attn_1_2_output):
        key = (attn_1_3_output, attn_1_2_output)
        if key in {
            (12, 10),
            (15, 2),
            (15, 10),
            (15, 12),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 8),
            (16, 9),
            (16, 10),
            (16, 11),
            (16, 12),
            (16, 14),
            (16, 15),
            (16, 16),
            (16, 17),
            (16, 18),
        }:
            return 10
        return 19

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_2_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output):
        key = num_attn_1_2_output
        return 7

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_2_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_1_output, num_attn_1_1_output):
        key = (num_attn_0_1_output, num_attn_1_1_output)
        return 8

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_2_output, num_attn_0_1_output):
        key = (num_attn_1_2_output, num_attn_0_1_output)
        return 16

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_1_output):
        key = num_attn_1_1_output
        if key in {0}:
            return 6
        return 8

    num_mlp_1_3_outputs = [num_mlp_1_3(k0) for k0 in num_attn_1_1_outputs]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_1_output, mlp_0_1_output):
        if attn_0_1_output in {"}", "(", "<s>", "{"}:
            return mlp_0_1_output == 2
        elif attn_0_1_output in {")"}:
            return mlp_0_1_output == 5

    attn_2_0_pattern = select_closest(mlp_0_1_outputs, attn_0_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_0_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, mlp_0_1_output):
        if position in {0, 10, 11, 12, 13, 14, 15, 17, 18, 19}:
            return mlp_0_1_output == 10
        elif position in {8, 1, 2, 16}:
            return mlp_0_1_output == 2
        elif position in {3, 4, 5, 6, 7, 9}:
            return mlp_0_1_output == 8

    attn_2_1_pattern = select_closest(mlp_0_1_outputs, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_0_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, mlp_0_0_output):
        if token in {"}", ")", "(", "<s>", "{"}:
            return mlp_0_0_output == 10

    attn_2_2_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_0_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return k_position == 1
        elif q_position in {8, 6, 7}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 7

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_1_output, mlp_0_1_output):
        if attn_0_1_output in {"}", "("}:
            return mlp_0_1_output == 2
        elif attn_0_1_output in {")", "<s>", "{"}:
            return mlp_0_1_output == 5

    num_attn_2_0_pattern = select(mlp_0_1_outputs, attn_0_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, mlp_0_0_output):
        if attn_1_1_output in {0, 5, 8, 11, 13, 14}:
            return mlp_0_0_output == 1
        elif attn_1_1_output in {1}:
            return mlp_0_0_output == 11
        elif attn_1_1_output in {9, 2, 17}:
            return mlp_0_0_output == 10
        elif attn_1_1_output in {3, 4, 7, 10, 19}:
            return mlp_0_0_output == 5
        elif attn_1_1_output in {6}:
            return mlp_0_0_output == 2
        elif attn_1_1_output in {18, 12}:
            return mlp_0_0_output == 8
        elif attn_1_1_output in {15}:
            return mlp_0_0_output == 12
        elif attn_1_1_output in {16}:
            return mlp_0_0_output == 19

    num_attn_2_1_pattern = select(mlp_0_0_outputs, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(q_mlp_0_2_output, k_mlp_0_2_output):
        if q_mlp_0_2_output in {0}:
            return k_mlp_0_2_output == 17
        elif q_mlp_0_2_output in {1, 14, 17, 18, 19}:
            return k_mlp_0_2_output == 11
        elif q_mlp_0_2_output in {2}:
            return k_mlp_0_2_output == 1
        elif q_mlp_0_2_output in {3, 12}:
            return k_mlp_0_2_output == 6
        elif q_mlp_0_2_output in {4, 7}:
            return k_mlp_0_2_output == 14
        elif q_mlp_0_2_output in {8, 5, 6}:
            return k_mlp_0_2_output == 13
        elif q_mlp_0_2_output in {9}:
            return k_mlp_0_2_output == 3
        elif q_mlp_0_2_output in {10}:
            return k_mlp_0_2_output == 5
        elif q_mlp_0_2_output in {11}:
            return k_mlp_0_2_output == 8
        elif q_mlp_0_2_output in {13}:
            return k_mlp_0_2_output == 2
        elif q_mlp_0_2_output in {16, 15}:
            return k_mlp_0_2_output == 12

    num_attn_2_2_pattern = select(mlp_0_2_outputs, mlp_0_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_3_output, token):
        if attn_1_3_output in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            18,
            19,
        }:
            return token == ""
        elif attn_1_3_output in {10}:
            return token == ")"
        elif attn_1_3_output in {17}:
            return token == "<pad>"

    num_attn_2_3_pattern = select(tokens, attn_1_3_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_1_2_output):
        key = mlp_1_2_output
        if key in {14}:
            return 10
        elif key in {4}:
            return 12
        return 14

    mlp_2_0_outputs = [mlp_2_0(k0) for k0 in mlp_1_2_outputs]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_1_output, position):
        key = (attn_2_1_output, position)
        if key in {
            (0, 5),
            (1, 5),
            (2, 1),
            (2, 2),
            (2, 5),
            (3, 1),
            (3, 5),
            (4, 5),
            (5, 5),
            (6, 5),
            (7, 1),
            (7, 2),
            (7, 5),
            (8, 5),
            (9, 5),
            (11, 5),
            (12, 5),
            (13, 5),
            (14, 5),
            (15, 5),
            (16, 1),
            (16, 5),
            (17, 5),
            (18, 5),
            (19, 5),
        }:
            return 6
        elif key in {(4, 1)}:
            return 11
        elif key in {(4, 3)}:
            return 13
        return 10

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_1_outputs, positions)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_1_3_output, attn_2_1_output):
        key = (attn_1_3_output, attn_2_1_output)
        if key in {
            (0, 2),
            (0, 5),
            (0, 9),
            (1, 1),
            (1, 2),
            (1, 5),
            (1, 8),
            (1, 9),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 8),
            (2, 9),
            (2, 11),
            (2, 13),
            (2, 14),
            (2, 16),
            (2, 18),
            (2, 19),
            (3, 2),
            (3, 5),
            (3, 8),
            (3, 9),
            (4, 5),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 13),
            (5, 14),
            (5, 16),
            (5, 18),
            (5, 19),
            (6, 2),
            (6, 5),
            (6, 9),
            (7, 2),
            (7, 5),
            (7, 9),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 16),
            (8, 17),
            (8, 18),
            (8, 19),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 8),
            (9, 9),
            (9, 13),
            (9, 14),
            (9, 16),
            (10, 5),
            (10, 9),
            (11, 2),
            (11, 5),
            (11, 8),
            (11, 9),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (13, 7),
            (13, 8),
            (13, 9),
            (13, 10),
            (13, 11),
            (13, 12),
            (13, 13),
            (13, 14),
            (13, 16),
            (13, 17),
            (13, 18),
            (13, 19),
            (14, 1),
            (14, 2),
            (14, 5),
            (14, 8),
            (14, 9),
            (17, 1),
            (17, 2),
            (17, 5),
            (17, 8),
            (17, 9),
            (18, 2),
            (18, 5),
            (18, 9),
            (19, 5),
        }:
            return 15
        elif key in {(10, 1), (10, 2), (10, 8)}:
            return 12
        return 13

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_2_1_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(num_mlp_1_1_output, attn_1_0_output):
        key = (num_mlp_1_1_output, attn_1_0_output)
        if key in {
            (0, 1),
            (0, 18),
            (0, 19),
            (1, 1),
            (1, 18),
            (1, 19),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 18),
            (2, 19),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (3, 17),
            (3, 18),
            (3, 19),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (4, 16),
            (4, 17),
            (4, 18),
            (4, 19),
            (5, 0),
            (5, 1),
            (5, 3),
            (5, 4),
            (5, 6),
            (5, 7),
            (5, 12),
            (5, 18),
            (5, 19),
            (6, 1),
            (6, 18),
            (6, 19),
            (7, 1),
            (7, 18),
            (7, 19),
            (9, 18),
            (9, 19),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 9),
            (10, 10),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (10, 17),
            (10, 18),
            (10, 19),
            (11, 1),
            (11, 11),
            (11, 18),
            (11, 19),
            (12, 1),
            (12, 18),
            (12, 19),
            (13, 1),
            (13, 18),
            (13, 19),
            (14, 1),
            (15, 1),
            (15, 2),
            (15, 5),
            (15, 18),
            (15, 19),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 8),
            (16, 9),
            (16, 10),
            (16, 11),
            (16, 12),
            (16, 13),
            (16, 14),
            (16, 15),
            (16, 17),
            (16, 18),
            (16, 19),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (17, 8),
            (17, 10),
            (17, 11),
            (17, 12),
            (17, 13),
            (17, 14),
            (17, 15),
            (17, 18),
            (17, 19),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 7),
            (18, 8),
            (18, 9),
            (18, 10),
            (18, 11),
            (18, 12),
            (18, 13),
            (18, 14),
            (18, 15),
            (18, 17),
            (18, 18),
            (18, 19),
            (19, 18),
            (19, 19),
        }:
            return 9
        return 18

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(num_mlp_1_1_outputs, attn_1_0_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_1_output, num_attn_1_3_output):
        key = (num_attn_2_1_output, num_attn_1_3_output)
        if key in {(56, 0), (57, 0), (58, 0), (59, 0)}:
            return 2
        return 13

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_1_output):
        key = num_attn_2_1_output
        if key in {0}:
            return 5
        return 9

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_2_1_outputs]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_0_2_output, num_attn_0_1_output):
        key = (num_attn_0_2_output, num_attn_0_1_output)
        if key in {
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (4, 1),
            (5, 0),
            (5, 1),
            (6, 0),
            (6, 1),
            (7, 0),
            (7, 1),
            (8, 0),
            (8, 1),
            (9, 0),
            (9, 1),
            (9, 2),
            (10, 0),
            (10, 1),
            (10, 2),
            (11, 0),
            (11, 1),
            (11, 2),
            (12, 0),
            (12, 1),
            (12, 2),
            (13, 0),
            (13, 1),
            (13, 2),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (24, 5),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 4),
            (25, 5),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (26, 5),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (27, 5),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (29, 6),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 5),
            (30, 6),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (31, 6),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (32, 5),
            (32, 6),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (34, 7),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (35, 7),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (36, 7),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (37, 7),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (39, 8),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (40, 8),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (41, 8),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (42, 8),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (43, 5),
            (43, 6),
            (43, 7),
            (43, 8),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (44, 5),
            (44, 6),
            (44, 7),
            (44, 8),
            (44, 9),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (45, 5),
            (45, 6),
            (45, 7),
            (45, 8),
            (45, 9),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (46, 5),
            (46, 6),
            (46, 7),
            (46, 8),
            (46, 9),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (47, 5),
            (47, 6),
            (47, 7),
            (47, 8),
            (47, 9),
            (48, 0),
            (48, 1),
            (48, 2),
            (48, 3),
            (48, 4),
            (48, 5),
            (48, 6),
            (48, 7),
            (48, 8),
            (48, 9),
            (49, 0),
            (49, 1),
            (49, 2),
            (49, 3),
            (49, 4),
            (49, 5),
            (49, 6),
            (49, 7),
            (49, 8),
            (49, 9),
            (49, 10),
            (50, 0),
            (50, 1),
            (50, 2),
            (50, 3),
            (50, 4),
            (50, 5),
            (50, 6),
            (50, 7),
            (50, 8),
            (50, 9),
            (50, 10),
            (51, 0),
            (51, 1),
            (51, 2),
            (51, 3),
            (51, 4),
            (51, 5),
            (51, 6),
            (51, 7),
            (51, 8),
            (51, 9),
            (51, 10),
            (52, 0),
            (52, 1),
            (52, 2),
            (52, 3),
            (52, 4),
            (52, 5),
            (52, 6),
            (52, 7),
            (52, 8),
            (52, 9),
            (52, 10),
            (53, 0),
            (53, 1),
            (53, 2),
            (53, 3),
            (53, 4),
            (53, 5),
            (53, 6),
            (53, 7),
            (53, 8),
            (53, 9),
            (53, 10),
            (54, 0),
            (54, 1),
            (54, 2),
            (54, 3),
            (54, 4),
            (54, 5),
            (54, 6),
            (54, 7),
            (54, 8),
            (54, 9),
            (54, 10),
            (54, 11),
            (55, 0),
            (55, 1),
            (55, 2),
            (55, 3),
            (55, 4),
            (55, 5),
            (55, 6),
            (55, 7),
            (55, 8),
            (55, 9),
            (55, 10),
            (55, 11),
            (56, 0),
            (56, 1),
            (56, 2),
            (56, 3),
            (56, 4),
            (56, 5),
            (56, 6),
            (56, 7),
            (56, 8),
            (56, 9),
            (56, 10),
            (56, 11),
            (57, 0),
            (57, 1),
            (57, 2),
            (57, 3),
            (57, 4),
            (57, 5),
            (57, 6),
            (57, 7),
            (57, 8),
            (57, 9),
            (57, 10),
            (57, 11),
            (58, 0),
            (58, 1),
            (58, 2),
            (58, 3),
            (58, 4),
            (58, 5),
            (58, 6),
            (58, 7),
            (58, 8),
            (58, 9),
            (58, 10),
            (58, 11),
            (59, 0),
            (59, 1),
            (59, 2),
            (59, 3),
            (59, 4),
            (59, 5),
            (59, 6),
            (59, 7),
            (59, 8),
            (59, 9),
            (59, 10),
            (59, 11),
            (59, 12),
        }:
            return 11
        return 6

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_0_output):
        key = num_attn_1_0_output
        return 9

    num_mlp_2_3_outputs = [num_mlp_2_3(k0) for k0 in num_attn_1_0_outputs]
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


print(run(["<s>", "{", "{", "(", "{", "}", ")", "}", "}", "("]))
