import json
from enum import StrEnum

from pydantic import BaseModel


class WeaponName(StrEnum):
    FIST = 'Fist',
    CHAINSAW = 'Chainsaw',
    PISTOL = 'Pistol',
    SHOTGUN = 'Shotgun',
    SUPER_SHOTGUN = 'SuperShotgun',
    CHAINGUN = 'Chaingun',
    ROCKET_LAUNCHER = 'RocketLauncher',
    PLASMA_RIFLE = 'Plasma Rifle',
    BFG9000 = 'BFG900',
    NONE = ''


class MonsterType(StrEnum):
    ZOMBIEMAN = 'Zombieman',
    SHOTGUN_GUY = 'ShotgunGuy',
    IMP = 'DoomImp'
    DEMON = 'Demon'
    SPECTRE = 'Spectre'
    LOST_SOUL = 'LostSoul'
    CACODEMON = 'Cacodemon'
    BARON_OF_HELL = 'BaronOfHell'
    CYBERDEMON = 'Cyberdemon'
    SPIDER_MASTERMIND = 'SpiderMastermind'


class AimedAtType(StrEnum):
    ACTOR = 'Actor'
    MONSTER = 'Monster'
    WALL = 'Wall'
    CEILING = 'Ceiling'
    FLOOR = 'Floor'
    UNKNOWN = 'Unknown'


class GroundCheckModel(BaseModel):
    isSprinting: bool
    terrainType: str
    obstacleDistance: float
    floorHeightAhead: float
    playerFloorHeight: float
    heightDifference: float
    isJumpable: bool
    isInAir: bool


class AimedAtModel(BaseModel):
    entityType: AimedAtType
    distance: float
    interactable: bool
    horizontalAngle: float
    verticalAngle: float


class MonsterModel(BaseModel):
    monsterType: MonsterType
    monsterMass: int
    monsterHealth: int
    distance: float
    relativeAngle: float
    relativePitch: float
    inFOV: bool
    screenX: float
    screenY: float


class InventorySlotModel(BaseModel):
    index: int
    weaponName: WeaponName
    ammoCount: int
    canUse: bool


class InventoryModel(BaseModel):
    currentSlot: int
    inventorySlots: list[InventorySlotModel]


class DoomGameState(BaseModel):
    AIMED_AT: AimedAtModel
    MONSTERS: list[MonsterModel]
    INVENTORY: InventoryModel
    GROUND_CHECK: GroundCheckModel

    # TODO: method to transform into prompt-ready format
    # TODO: Potentially abstract GameState class?
    def to_prompt_ready(self) -> str:
        d = self.model_dump()
        del d['GROUND_CHECK']
        del d['AIMED_AT']['horizontalAngle']
        del d['AIMED_AT']['verticalAngle']
        for m in d['MONSTERS']:
            del m['monsterMass']
            del m['inFOV']
            del m['screenX']
            del m['screenY']
        d['INVENTORY']['inventorySlots'] = [
            w
            for w in d['INVENTORY']['inventorySlots']
            if w['canUse']
        ]
        for w in d['INVENTORY']['inventorySlots']:
            del w['canUse']
        return json.dumps(d)