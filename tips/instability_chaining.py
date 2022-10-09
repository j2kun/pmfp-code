"""An implementation of the student-proposing instability chaining algorithm."""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import cast
import heapq
import logging


T = TypeVar("T")


def precedes(L, item1, item2):
    return L.index(item1) < L.index(item2)


def split_by(lst, f):
    trues = []
    falses = []
    for x in lst:
        if f(x):
            trues.append(x)
        else:
            falses.append(x)
    return trues, falses


class Applicant(ABC):
    @abstractmethod
    def prefers(
        self,
        programs: Tuple["ResidencyProgram", Optional["ResidencyProgram"]],
        matching: "Matching",
    ) -> bool:
        """Return True if the applicant prefers the input program(s) over its match."""
        ...

    @abstractmethod
    def reset_best(self) -> None:
        """Reset the order of programs to apply to."""
        ...

    @abstractmethod
    def map(self, fn: Callable[["Student"], T]) -> Tuple[T, Optional[T]]:
        """Map a function over the members in the Applicant."""
        ...

    @abstractmethod
    def may_still_apply(self) -> bool:
        """Return True if there are still more programs to apply to."""
        ...

    def proposal_pair(self) -> Tuple["Student", Optional["Student"]]:
        """Return the 1-2 students that will apply to schools jointly."""
        return self.map(lambda s: s)

    def is_single(self) -> bool:
        return self.proposal_pair()[1] is None


@dataclass
class Student(Applicant):
    id: int

    """Preferences on ResidencyProgram.id, from highest priority to lowest priority."""
    preferences: List[int]

    """The highest priority program this student has yet to be rejected from."""
    best_unrejected: int = 0

    def to_apply(self):
        return self.preferences[self.best_unrejected]

    def may_still_apply(self) -> bool:
        return self.best_unrejected < len(self.preferences)

    def prefers(
        self,
        programs: Tuple["ResidencyProgram", Optional["ResidencyProgram"]],
        matching: "Matching",
    ) -> bool:
        program, _ = programs
        if (
            self not in matching.matches
            or matching.matches[self].id not in self.preferences  # weird
        ):
            return program.id in self.preferences

        try:
            # This implies that a student cannot prefer a program that they are
            # already assigned to.
            return precedes(self.preferences, program.id, matching.matches[self].id)
        except ValueError:
            # If the student doesn't have the program in their list, treat as
            # a non-preference.
            return False

    def reset_best(self):
        self.best_unrejected = 0

    def map(self, fn):
        return (fn(self), None)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, Student) and self.id == other.id

    def __str__(self):
        return f"Student({self.id})"

    def __lt__(self, other):
        return self.id < other.id


@dataclass
class Couple(Applicant):
    """A datatype representing a couple, whose preferences are considered jointly."""

    members: Tuple[Student, Student]

    def joint_preferences(self) -> List[Tuple[int, int]]:
        """Return the list of joint preferences of the couple."""
        return list(zip(self.members[0].preferences, self.members[1].preferences))

    @property
    def best_unrejected(self):
        # Preference order is synchronized, so can always use the first member.
        return self.members[0].best_unrejected

    def may_still_apply(self) -> bool:
        return self.best_unrejected < len(self.members[0].preferences)

    def prefers(
        self,
        programs: Tuple["ResidencyProgram", Optional["ResidencyProgram"]],
        matching: "Matching",
    ) -> bool:
        s1, s2 = self.members
        p1, p2 = programs
        assert p2 is not None
        m1, m2 = matching.current_match(self)
        if m1 is None and m2 is None:
            return p1.id in s1.preferences and p2.id in p2.preferences
        elif m2 is None:
            return p2.id in s2.preferences and s1.prefers((p1, None), matching)
        elif m1 is None:
            return p1.id in s1.preferences and s2.prefers((p2, None), matching)

        # The two members must both jointly prefer the program pair over their
        # current matches.
        joint_prefs = self.joint_preferences()
        try:
            return precedes(joint_prefs, (p1.id, p2.id), (m1.id, m2.id))
        except ValueError:
            # If the student doesn't have (p1, p2) in their list, treat as a
            # non-preference.
            return False

    def reset_best(self):
        for member in self.members:
            member.reset_best()

    def map(self, fn):
        return tuple(fn(m) for m in self.members)

    def __post_init__(self):
        self.members = tuple(sorted(self.members))
        assert len(self.members[0].preferences) == len(self.members[1].preferences)

    def __str__(self):
        return f"({self.members[0]}, {self.members[1]})"

    def __eq__(self, other):
        return isinstance(other, Couple) and self.members == other.members

    def __hash__(self):
        return hash(self.members)


@dataclass
class ResidencyProgram:
    id: int

    """Preferences on Student.id, from highest priority to lowest priority."""
    preferences: List[int]

    """The number of open spots."""
    capacity: int

    def select(self, pool: Set[Student]) -> Set[Student]:
        """Select students from `pool` by priority. Return unchosen students."""
        chosen = heapq.nsmallest(
            self.capacity, pool, key=lambda s: self.preferences.index(s.id)
        )
        return pool - set(chosen)

    def prefers(self, matching: "Matching", *students: Student) -> bool:
        """Return True if program prefers all input students to its match in `matching`."""
        student_set = set(students)
        displ = self.select(student_set | matching.students_matched_to(self))
        return not (student_set & displ)

    def __hash__(self):
        return self.id

    def __str__(self):
        return f"Program({self.id})"

    def __eq__(self, other):
        return isinstance(other, ResidencyProgram) and self.id == other.id


class ProgramIndex:
    def __init__(self, programs: Iterable[ResidencyProgram]):
        self.index = {program.id: program for program in programs}

    def __getitem__(self, program_id: int) -> ResidencyProgram:
        return self.index[program_id]

    def __iter__(self):
        return iter(self.index.values())

    def next_to_apply(
        self, applicant: Applicant
    ) -> Tuple[ResidencyProgram, Optional[ResidencyProgram]]:
        if isinstance(applicant, Student):
            return (self.index[applicant.to_apply()], None)
        else:
            applicant = cast(Couple, applicant)
            return (
                self.index[applicant.members[0].to_apply()],
                self.index[applicant.members[1].to_apply()],
            )


@dataclass
class Matching:
    matches: Dict[Student, ResidencyProgram]
    # For simplicity in this Tip, we assume there is no unassigned bucket.

    applicants: List[Applicant]
    programs: List[ResidencyProgram]
    valid: bool = True

    def set(
        self,
        *matches: Tuple[Optional[Student], Optional[ResidencyProgram]],
    ) -> None:
        for (student, program) in matches:
            if student and program:
                logging.debug(f"Matching {student} to {program}")
                self.matches[student] = program

    def current_match(
        self, applicant: Applicant
    ) -> Tuple[Optional[ResidencyProgram], Optional[ResidencyProgram]]:
        if isinstance(applicant, Student):
            return (self.matches[applicant], None)
        else:
            applicant = cast(Couple, applicant)
            return (
                self.matches.get(applicant.members[0]),
                self.matches.get(applicant.members[1]),
            )

    def reject(self, student):
        if student in self.matches:
            del self.matches[student]
            student.best_unrejected += 1

    def students_matched_to(self, program: ResidencyProgram) -> Set[Student]:
        # Inefficient, but let's keep it simple.
        return set(s for (s, prog) in self.matches.items() if prog.id == program.id)

    def make_pool(self, program_index: ProgramIndex, student: Student) -> Set[Student]:
        program, _ = program_index.next_to_apply(student)
        return set([student]) | self.students_matched_to(program)

    def __str__(self):
        return "\n".join(
            [f"{student} -> {program}" for (student, program) in self.matches.items()]
        )


class InstabilityChaining:
    def __init__(
        self,
        applicants: Iterable[Applicant],
        programs: Iterable[ResidencyProgram],
    ):
        self.program_index = ProgramIndex(programs)

        couples = set(x for x in applicants if isinstance(x, Couple))
        partner_mapping = dict(c.members for c in couples)
        self.partner_mapping = partner_mapping | dict(
            (v, k) for k, v in partner_mapping.items()
        )

        # Clean up dupes if a couple and its members are in applicants.
        singles: List[Applicant] = [
            x
            for x in applicants
            if isinstance(x, Student) and x not in self.partner_mapping
        ]

        # Processing couples last reduces the chance of cycles
        self.applicants: List[Applicant] = singles + list(couples)
        self.matching = Matching(
            matches=dict(), applicants=self.applicants, programs=list(programs)
        )

        # a log used to detect cycles
        self.log: Set[Any] = set()

    def run(self) -> Matching:
        try:
            for applicant in self.applicants:
                self.process_one(applicant)
        except ValueError:
            self.matching.valid = False

        return self.matching

    def process_one(self, applicant: Applicant) -> None:
        self.applicant_stack = list([applicant])
        self.program_stack: Set[ResidencyProgram] = set()

        # reset the log after processing each applicant
        self.log = set()

        while self.applicant_stack or self.program_stack:
            while self.applicant_stack:
                # This could be improved to ensure a couple is always selected
                # over a single student, if a couple is in the stack.
                self.apply(self.applicant_stack.pop())

            if self.program_stack:
                program = self.program_stack.pop()
                logging.debug(f"Processing {program} from the program stack.")
                unstable_applicants = unstable_pairs(
                    program, self.matching, self.program_index
                )
                for applicant in unstable_applicants:
                    applicant.reset_best()
                    # By resetting their proposals back to the start of their
                    # preference lists, they effectively withdraw from their
                    # current program, and because they may end up matched with
                    # a higher-ranked program, that can introduce a further
                    # unstable pair with the newly vacant position.
                    m1, m2 = self.matching.current_match(applicant)
                    logging.debug(
                        f"Adding potentially withdrawn programs {m1}, {m2} "
                        "to the program stack."
                    )
                    assert m1 is not None
                    self.program_stack.add(m1)
                    if m2:
                        self.program_stack.add(m2)

                if unstable_applicants:
                    logging.debug(
                        f"Adding unstable applicants "
                        f"{','.join(str(x) for x in unstable_applicants)} "
                        f"to the applicant stack"
                    )
                self.applicant_stack.extend(unstable_applicants)

            self.snapshot()

    def snapshot(self) -> None:
        hashable = (
            tuple(hash(x) for x in self.applicant_stack),
            tuple(hash(x) for x in self.program_stack),
            tuple(sorted((s.id, p.id) for (s, p) in self.matching.matches.items())),
        )
        if hashable in self.log:
            logging.debug(
                f"Found a cycle: log={self.log}\nNext entry would be {hashable}"
            )
            raise ValueError("Found a cycle")
        self.log.add(hashable)

    def apply(self, applicant: Applicant) -> None:
        logging.debug(f"{applicant} starts applying")
        while applicant.may_still_apply():
            proposer, partner = applicant.proposal_pair()
            program, partner_program = self.program_index.next_to_apply(applicant)
            applicants = self.matching.make_pool(self.program_index, proposer)

            # A special case if both partners prefer the same program. The program
            # needs to prefer both partners and bump two held applications.
            if partner and program == partner_program:
                applicants.add(partner)

            logging.debug(f"{applicant} proposing to {program}, {partner_program}")
            displ = program.select(applicants)
            if partner and program != partner_program:
                assert partner_program is not None
                displ |= partner_program.select(
                    self.matching.make_pool(self.program_index, partner)
                )

            if not displ:
                logging.debug("Application accepted with no bumps")
                # The program(s) applied to had available positions and didn't
                # need to reject anyone.
                self.matching.set((proposer, program), (partner, partner_program))
                break

            rejected = proposer in displ or partner in displ

            if rejected:
                rejectee = f"{proposer if proposer in displ else partner}"
                rejecter = f"{program if proposer in displ else partner_program}"
                logging.debug(
                    f"Application rejected ({rejectee} rejected by {rejecter})"
                )
                proposer.best_unrejected += 1
                if partner:
                    partner.best_unrejected += 1
                continue
            else:
                logging.debug("Application accepted with bumps")

            self.matching.set((proposer, program), (partner, partner_program))

            # Not being rejected means that we have to handle all the displaced applicants.
            # This can come in two cases:
            #
            #  - If all displaced applicants are single, continue with one such person as
            #    the new applicant, put the remaining applicants on the applicant stack.
            #
            #  - Each displaced member of a couple has their partner withdraw from their
            #    matched program (if any) and the program is added to the program stack.
            #    Pick one such couple and continue with them applying jointly down their
            #    preference lists, and put the remaining couples on the applicant stack.
            couples, singles = split_by(displ, lambda s: s in self.partner_mapping)
            for bumped in singles:
                logging.debug(
                    f"{bumped} (matched to {self.matching.matches[bumped]}) "
                    f"displaced by {applicant}"
                )
                self.matching.reject(bumped)

            if len(singles) == len(displ):
                applicant = displ.pop()
                self.applicant_stack.extend(displ)
                continue

            displaced_couples: Set[Couple] = set()
            for bumped in couples:
                withdrawer = self.partner_mapping[bumped]
                couple = Couple(members=(bumped, withdrawer))
                if couple in displaced_couples:
                    continue
                logging.debug(
                    f"{couple} displaced (was matched to "
                    f"{self.matching.matches[bumped]}, "
                    f"{self.matching.matches[withdrawer]}) "
                    f"by {applicant}\n"
                    f"{withdrawer} withdrawing from {self.matching.matches[withdrawer]}\n"
                    f"Adding withdrawn program {self.matching.matches[withdrawer]} to "
                    "the program stack"
                )
                displaced_couples.add(couple)
                self.program_stack.add(self.matching.matches[withdrawer])
                self.matching.reject(bumped)
                self.matching.reject(withdrawer)

            applicant = displaced_couples.pop()
            logging.debug(
                f"Adding displaced couples {displaced_couples} and singles {singles} "
                f"to the applicant stack"
            )
            self.applicant_stack.extend(displaced_couples)
            self.applicant_stack.extend(singles)


def stable_matching(
    applicants: Iterable[Applicant],
    programs: Iterable[ResidencyProgram],
) -> Matching:
    """Construct a stable matching of students to residency programs.

    Implemented as a variation of the instability chaining algorithm from

        Roth, Alvin E. and Vande Vate, John H.
        'Random Paths to Stability in Two-Sided Matching.'
        Econometrica, November 1990, 58(6), pp. 1475–80.

    Note singles and couples require unique ids.

    Arguments:
        single_students: the students to match
        couples: the couples to match
        programs: the ResidencyPrograms to match

    Returns:
        A dict {Student: ResidencyProgram} assigning each student to a program.
    """
    return InstabilityChaining(applicants, programs).run()


def unstable_pairs(
    program: ResidencyProgram,
    matching: Matching,
    index: ProgramIndex,
) -> Set[Applicant]:
    """Returns all applicants in the market that are unstable with `program`."""
    unstable_applicants: Set[Applicant] = set()

    for applicant in matching.applicants:
        s1, s2 = applicant.proposal_pair()
        if s1 not in matching.matches:
            # It would be considered "unstable" by default, which can result in
            # this applicant being incorrectly added to the applicant stack
            # without an existing match.
            continue

        program_prefs = applicant.map(lambda s: program.prefers(matching, s))
        program_has_pref = all(True if x is None else x for x in program_prefs)
        if program_has_pref:
            if applicant.is_single():
                if applicant.prefers((program, None), matching):
                    unstable_applicants.add(applicant)
            else:
                couple = cast(Couple, applicant)
                s1, s2 = couple.members
                for (p1_id, p2_id) in couple.joint_preferences():
                    p1, p2 = index[p1_id], index[p2_id]
                    if p1 != program and p2 != program:
                        continue

                    applicant_pref = couple.prefers((p1, p2), matching)
                    if p1 == p2:
                        # The program must have room and preference for both
                        program_pref = p1.prefers(matching, *couple.members)
                    else:
                        program_pref = p1.prefers(matching, s1) and p2.prefers(
                            matching, s2
                        )

                    if applicant_pref and program_pref:
                        unstable_applicants.add(couple)
                        break

    return unstable_applicants


def find_unstable_pairs(matching: Matching) -> List[Tuple[Applicant, ResidencyProgram]]:
    program_index = ProgramIndex(matching.programs)
    return [
        (app, prog)
        for prog in program_index
        for app in unstable_pairs(prog, matching, program_index)
    ]
