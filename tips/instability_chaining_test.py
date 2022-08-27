import math

from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from hypothesis.strategies import permutations

from instability_chaining import Couple
from instability_chaining import Matching
from instability_chaining import ResidencyProgram
from instability_chaining import Student
from instability_chaining import stable_matching
from instability_chaining import find_unstable_pairs


def build_partner_mapping(couples):
    partner_mapping = dict(couple.members for couple in couples)
    return partner_mapping | dict((v, k) for k, v in partner_mapping.items())


@composite
def market(
    draw,
    num_students=integers(min_value=2, max_value=5),
    num_programs=integers(min_value=1, max_value=5),
    program_capacity=integers(min_value=1, max_value=5),
    include_couples=True,
):
    """A hypothesis rule to generate a random matching market."""
    student_ids = list(range(draw(num_students)))
    program_ids = list(range(draw(num_programs)))

    min_capacity = int(math.ceil(len(student_ids) / len(program_ids)))

    students = []
    programs = []

    for student_id in student_ids:
        preferences = draw(permutations(program_ids))
        students.append(
            Student(
                id=student_id,
                preferences=preferences,
                best_unrejected=0,
            )
        )

    student_index = {s.id: s for s in students}

    for program_id in program_ids:
        preferences = draw(permutations(student_ids))
        programs.append(
            ResidencyProgram(
                id=program_id,
                preferences=preferences,
                capacity=max(min_capacity, draw(program_capacity)),
            )
        )

    couples = []
    if include_couples:
        num_couples = draw(integers(min_value=1, max_value=len(student_ids) // 2))
        couplings = draw(permutations(student_ids))
        for i in range(num_couples):
            s1 = student_index[couplings[2 * i]]
            s2 = student_index[couplings[2 * i + 1]]
            couples.append(Couple(members=(s1, s2)))

    partner_mapping = build_partner_mapping(couples)
    single_students = [s for s in students if s not in partner_mapping]

    return (single_students, couples, programs, partner_mapping)


def err_msg(matching, unstable_pair):
    applicant, program = unstable_pair
    if isinstance(applicant, Student):
        (student, partner) = (applicant, None)
    elif isinstance(applicant, Couple):
        (student, partner) = applicant.members

    assigned_to_program = [
        str(x) for (x, y) in matching.matches.items() if y == program
    ]
    student_assigned = matching.matches[student]
    msg = (
        f"\n({applicant}, {program}) form an unstable pair in matching \n\n{matching} "
        f"\n\nbecause {student} prefers {program} to their assignment {student_assigned}"
        f"\nand {program} prefers {student} over at least one of {assigned_to_program}.\n"
    )

    if partner:
        partner_assigned = matching.matches[partner]
        partner_program = partner.preferences[student.preferences.index(program.id)]
        assigned_to_program = [
            str(x) for (x, y) in matching.matches.items() if y.id == partner_program
        ]
        msg += (
            f"And {student}'s partner {partner} simultaneously prefers Program({partner_program}) "
            f"over their assignment {partner_assigned}\nand Program({partner_program}) prefers "
            f"{partner} over at least one of {assigned_to_program}.\n"
        )

    return msg


def assert_stable(programs, matching, partner_mapping=None):
    partner_mapping = partner_mapping or dict()
    unstable_pairs = find_unstable_pairs(programs, matching, partner_mapping)
    assert unstable_pairs == [], err_msg(matching, next(iter(unstable_pairs)))


"""
TODO: test cases

 - Test with a cycle

TODO: Cycle detection & randomization

TODO: re-read algorithm paper https://web.stanford.edu/~alroth/papers/rothperansonaer.PDF
to make sure I'm not missing anything.
"""


def test_unstable_pairs_stable():
    students = [Student(id=0, preferences=[0, 1]), Student(id=1, preferences=[1, 0])]
    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[0, 1]),
        ResidencyProgram(id=1, capacity=1, preferences=[1, 0]),
    ]
    matching = Matching(matches={students[0]: programs[0], students[1]: programs[1]})
    assert find_unstable_pairs(programs, matching, dict()) == []


def test_unstable_pairs_unstable():
    students = [Student(id=0, preferences=[1, 0]), Student(id=1, preferences=[1, 0])]
    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[0, 1]),
        ResidencyProgram(id=1, capacity=1, preferences=[0, 1]),
    ]
    matching = Matching(matches={students[0]: programs[0], students[1]: programs[1]})
    result = find_unstable_pairs(programs, matching, dict())
    assert len(result) == 1
    assert result[0] == (students[0], programs[1])


def test_unstable_pairs_stable_with_couple():
    students = [
        Student(id=0, preferences=[0, 2, 1]),
        Student(id=1, preferences=[1, 0, 2]),
        Student(id=2, preferences=[0, 1, 2]),  # 2 displaces 0
    ]
    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[2, 0, 1]),
        ResidencyProgram(id=1, capacity=1, preferences=[1, 0, 2]),
        ResidencyProgram(id=2, capacity=1, preferences=[1, 0, 2]),
    ]
    partner_mapping = {
        students[0]: students[1],
        students[1]: students[0],
    }
    matching = Matching(
        matches={
            students[0]: programs[1],
            students[1]: programs[2],
            students[2]: programs[0],
        }
    )
    assert find_unstable_pairs(programs, matching, partner_mapping) == []


"""
    Tests that involve no couples
"""


def test_stable_matching_two():
    students = [Student(id=0, preferences=[0, 1]), Student(id=1, preferences=[1, 0])]
    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[0, 1]),
        ResidencyProgram(id=1, capacity=1, preferences=[1, 0]),
    ]
    matching = stable_matching(students, [], programs)
    assert matching.matches == {students[0]: programs[0], students[1]: programs[1]}
    assert find_unstable_pairs(programs, matching, dict()) == []


def test_stable_matching_six():
    students = [
        Student(id=0, preferences=[3, 5, 4, 2, 1, 0]),
        Student(id=1, preferences=[2, 3, 1, 0, 4, 5]),
        Student(id=2, preferences=[5, 2, 1, 0, 3, 4]),
        Student(id=3, preferences=[0, 1, 2, 3, 4, 5]),
        Student(id=4, preferences=[4, 5, 1, 2, 0, 3]),
        Student(id=5, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[3, 5, 4, 2, 1, 0]),
        ResidencyProgram(id=1, capacity=1, preferences=[2, 3, 1, 0, 4, 5]),
        ResidencyProgram(id=2, capacity=1, preferences=[5, 2, 1, 0, 3, 4]),
        ResidencyProgram(id=3, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=4, capacity=1, preferences=[4, 5, 1, 2, 0, 3]),
        ResidencyProgram(id=5, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    matching = stable_matching(students, [], programs)
    assert matching.matches == {
        students[0]: programs[3],
        students[1]: programs[2],
        students[2]: programs[5],
        students[3]: programs[0],
        students[4]: programs[4],
        students[5]: programs[1],
    }
    assert find_unstable_pairs(programs, matching, dict()) == []


def test_stable_matching_all_tied():
    students = [
        Student(id=0, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=1, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=2, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=3, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=4, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=5, preferences=[5, 4, 3, 2, 1, 0]),
    ]

    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=1, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=2, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=3, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=4, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=5, capacity=1, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    matching = stable_matching(students, [], programs)
    assert matching.matches == {
        students[0]: programs[5],
        students[1]: programs[4],
        students[2]: programs[3],
        students[3]: programs[2],
        students[4]: programs[1],
        students[5]: programs[0],
    }
    assert find_unstable_pairs(programs, matching, dict()) == []


def test_stable_matching_capacity_2():
    students = [
        Student(id=0, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=1, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=2, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=3, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=4, preferences=[5, 4, 3, 2, 1, 0]),
        Student(id=5, preferences=[5, 4, 3, 2, 1, 0]),
    ]

    programs = [
        ResidencyProgram(id=0, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=1, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=2, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=3, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=4, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
        ResidencyProgram(id=5, capacity=2, preferences=[0, 1, 2, 3, 4, 5]),
    ]

    matching = stable_matching(students, [], programs)
    assert matching.matches == {
        students[0]: programs[5],
        students[1]: programs[5],
        students[2]: programs[4],
        students[3]: programs[4],
        students[4]: programs[3],
        students[5]: programs[3],
    }
    assert find_unstable_pairs(programs, matching, dict()) == []


@given(market(include_couples=False))
@settings(print_blob=True)
def test_stability_with_no_couples(students_and_programs):
    students, couples, programs, partner_mapping = students_and_programs
    couples = []
    matching = stable_matching(students, couples, programs)
    assert_stable(programs, matching)


"""
    Tests that involve couples
"""


def test_one_couple_not_displaced():
    students = [
        Student(id=0, preferences=[0, 1, 2]),
        Student(id=1, preferences=[1, 2, 0]),
        Student(id=2, preferences=[0, 1, 2]),  # The couple displaces the single student
    ]

    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[1, 0, 2]),
        ResidencyProgram(id=1, capacity=1, preferences=[1, 0, 2]),
        ResidencyProgram(id=2, capacity=1, preferences=[1, 0, 2]),
    ]

    couples = [Couple(members=(students[0], students[1]))]

    matching = stable_matching(students, couples, programs)
    assert matching.matches == {
        students[0]: programs[0],
        students[1]: programs[1],
        students[2]: programs[2],
    }
    assert find_unstable_pairs(programs, matching, dict()) == []


def test_one_couple_second_choice():
    students = [
        Student(id=0, preferences=[0, 2, 1]),
        Student(id=1, preferences=[1, 0, 2]),
        Student(id=2, preferences=[0, 1, 2]),  # 2 beats 0
    ]

    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[2, 0, 1]),
        ResidencyProgram(id=1, capacity=1, preferences=[1, 0, 2]),
        ResidencyProgram(id=2, capacity=1, preferences=[1, 0, 2]),
    ]

    couples = [Couple(members=(students[0], students[1]))]
    partner_mapping = build_partner_mapping(couples)
    single_students = [students[2]]

    matching = stable_matching(single_students, couples, programs)
    assert matching.matches == {
        students[0]: programs[1],
        students[1]: programs[2],
        students[2]: programs[0],
    }
    assert_stable(programs, matching, partner_mapping)


def test_one_couple_with_repeating_joint_preferences():
    # A test where the couple's joint preferences have a cycle when projecting
    # to just one member.
    students = [
        Student(id=0, preferences=[0, 1, 0, 1, 2, 2]),
        Student(id=1, preferences=[1, 1, 0, 0, 1, 2]),
        Student(id=2, preferences=[0, 1, 2]),  # 2 gets the spot at program 0
    ]

    # The couple has to go down to their second to last preference of (2, 1).
    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[2, 0, 1]),
        ResidencyProgram(id=1, capacity=1, preferences=[1, 0, 2]),
        ResidencyProgram(id=2, capacity=1, preferences=[1, 0, 2]),
    ]

    couples = [Couple(members=(students[0], students[1]))]
    partner_mapping = build_partner_mapping(couples)
    single_students = [students[2]]

    matching = stable_matching(single_students, couples, programs)
    assert matching.matches == {
        students[0]: programs[2],
        students[1]: programs[1],
        students[2]: programs[0],
    }
    assert_stable(programs, matching, partner_mapping)


def test_couple_displaces_entire_second_couple():
    # A test where one couple displaces another in both members.
    students = [
        Student(id=0, preferences=[0, 1, 0, 1, 2]),
        Student(id=1, preferences=[1, 1, 0, 0, 2]),
        Student(id=2, preferences=[0]),
        Student(id=3, preferences=[1]),
    ]

    # first couple (0, 1) gets (0, 1). Then second couple displaces both,
    # and the first couple has to go all the way to last preference.

    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[2, 3, 0, 1]),
        ResidencyProgram(id=1, capacity=1, preferences=[3, 2, 1, 0]),
        # lots of space over here at 2
        ResidencyProgram(id=2, capacity=4, preferences=[0, 1, 2, 3]),
    ]

    couples = [
        Couple(members=(students[0], students[1])),
        Couple(members=(students[2], students[3])),
    ]
    partner_mapping = build_partner_mapping(couples)
    single_students = []

    matching = stable_matching(single_students, couples, programs)
    assert matching.matches == {
        students[0]: programs[2],
        students[1]: programs[2],
        students[2]: programs[0],
        students[3]: programs[1],
    }
    assert_stable(programs, matching, partner_mapping)


def test_couple_displaces_first_member_of_second_couple():
    # A test where one couple displaces another in both members.
    students = [
        Student(id=0, preferences=[0, 1]),
        Student(id=1, preferences=[1, 2]),
        Student(id=2, preferences=[0]),
        Student(id=3, preferences=[2]),
    ]

    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[2, 3, 0, 1]),
        # Because program 1 favors student 1 the most, the only way
        # student 1 leaves program 1 is by withdrawing.
        ResidencyProgram(id=1, capacity=1, preferences=[1, 2, 3, 0]),
        ResidencyProgram(id=2, capacity=4, preferences=[0, 1, 2, 3]),
    ]

    couples = [
        Couple(members=(students[0], students[1])),
        Couple(members=(students[2], students[3])),
    ]
    partner_mapping = build_partner_mapping(couples)
    single_students = []

    matching = stable_matching(single_students, couples, programs)
    assert matching.matches == {
        students[0]: programs[1],
        students[1]: programs[2],
        students[2]: programs[0],
        students[3]: programs[2],
    }
    assert_stable(programs, matching, partner_mapping)


def test_couple_displaces_two_singles():
    students = [
        Student(id=0, preferences=[0]),
        Student(id=1, preferences=[1]),
        Student(id=2, preferences=[0, 1, 2]),
        Student(id=3, preferences=[1, 0, 2]),
    ]

    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[0, 2, 3, 1]),
        ResidencyProgram(id=1, capacity=1, preferences=[1, 3, 2, 0]),
        ResidencyProgram(id=2, capacity=4, preferences=[0, 1, 2, 3]),
    ]

    couples = [
        Couple(members=(students[0], students[1])),
    ]
    partner_mapping = build_partner_mapping(couples)
    single_students = [students[2], students[3]]

    matching = stable_matching(single_students, couples, programs)
    assert matching.matches == {
        students[0]: programs[0],
        students[1]: programs[1],
        students[2]: programs[2],
        students[3]: programs[2],
    }
    assert_stable(programs, matching, partner_mapping)


def test_couple_applies_to_same_program():
    students = [
        Student(id=0, preferences=[0]),
        Student(id=1, preferences=[0]),
        Student(id=2, preferences=[0, 1, 2]),
        Student(id=3, preferences=[0, 1, 2]),
    ]

    programs = [
        ResidencyProgram(id=0, capacity=2, preferences=[0, 1, 2, 3]),
        ResidencyProgram(id=1, capacity=1, preferences=[1, 3, 2, 0]),
        ResidencyProgram(id=2, capacity=4, preferences=[0, 1, 2, 3]),
    ]

    couples = [
        Couple(members=(students[0], students[1])),
    ]
    partner_mapping = build_partner_mapping(couples)
    single_students = [students[2], students[3]]

    matching = stable_matching(single_students, couples, programs)
    assert matching.matches == {
        students[0]: programs[0],
        students[1]: programs[0],
        students[2]: programs[2],
        students[3]: programs[1],
    }
    assert_stable(programs, matching, partner_mapping)


def test_withdrawal_creates_unstable_pair():
    # Scenario is:
    #
    # - student 0 starts at their mutual #1 pick, program 0.
    # - student 1, partner of student 0, gets bumped by couple (3, 4).
    # - couple (0, 1) both end up at program 2.
    # - student 2 is next pick for program 0, and student 2 prefers program 0
    #     but was re-assigned to program 2 after student 0 bumped them.
    # - algorithm must put program 0 on the stack, detect student 2 forms an
    #     unstable pair, then re-run deferred acceptance for student 2.
    students = [
        Student(id=0, preferences=[0, 2, 1]),
        Student(id=1, preferences=[1, 2, 0]),
        Student(id=2, preferences=[0, 2, 1]),
        # couple (3, 4) causes 1 to get bumped, 0 to withdraw.
        Student(id=3, preferences=[1, 2, 0]),
        Student(id=4, preferences=[2, 1, 0]),
    ]

    programs = [
        ResidencyProgram(id=0, capacity=1, preferences=[0, 2, 1, 3, 4]),
        ResidencyProgram(id=1, capacity=1, preferences=[3, 1, 2, 0, 4]),
        ResidencyProgram(id=2, capacity=4, preferences=[4, 1, 2, 3, 0]),
    ]

    couples = [
        Couple(members=(students[0], students[1])),
        Couple(members=(students[3], students[4])),
    ]
    partner_mapping = build_partner_mapping(couples)
    single_students = [students[2]]

    matching = stable_matching(single_students, couples, programs)
    assert matching.matches == {
        students[0]: programs[2],
        students[1]: programs[2],
        students[2]: programs[0],
        students[3]: programs[1],
        students[4]: programs[2],
    }
    assert_stable(programs, matching, partner_mapping)


@given(market(include_couples=True))
@settings(print_blob=True)
def test_stability_with_couples(students_and_programs):
    students, couples, programs, partner_mapping = students_and_programs
    matching = stable_matching(students, couples, programs)
    assert_stable(programs, matching, partner_mapping=partner_mapping)
