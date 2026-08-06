"""
Pydantic schemas and enums for project trend analytics extraction with fuzzy enum coercion.
"""
from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class EngineeringFieldEnum(str, Enum):
    CIVIL_INFRASTRUCTURE = "Civil & Infrastructure"
    WATER_SANITATION = "Water & Sanitation"
    ENERGY_POWER = "Energy & Power"
    TRANSPORTATION_ROADS = "Transportation & Roads"
    ENVIRONMENTAL = "Environmental & Natural Resources"
    BUILDINGS_ARCHITECTURE = "Buildings & Facilities"
    TELECOM_IT = "Telecom & IT"
    INDUSTRIAL_MINING = "Industrial & Mining"
    OTHER = "Other"


class ProjectTypeEnum(str, Enum):
    CONSTRUCTION_SUPERVISION = "Construction Supervision & Inspection"
    FEASIBILITY_DESIGN = "Feasibility, Audit & Design"
    MAINTENANCE_OPERATION = "Maintenance & Operation"
    TECHNICAL_CONSULTING = "Technical & Strategic Consulting"
    TURNKEY_EPC = "Turnkey / EPC Construction"
    OTHER = "Other"


class BudgetScaleEnum(str, Enum):
    MICRO = "< $500K"
    SMALL = "$500K - $2M"
    MEDIUM = "$2M - $10M"
    LARGE = "$10M - $50M"
    MEGA = "> $50M"
    UNDISCLOSED = "Undisclosed / Not Stated"


class FundingSourceEnum(str, Enum):
    PUBLIC_STATE = "Public State Budget"
    MULTILATERAL_BANK = "Multilateral Development Bank (IDB, World Bank, CAF, CABEI)"
    PRIVATE = "Private Capital"
    MIXED = "Public-Private Partnership (PPP)"
    UNKNOWN = "Unknown"


class ProjectTrendExtraction(BaseModel):
    engineering_field: EngineeringFieldEnum = Field(
        ..., description="Primary engineering discipline or sector of the project"
    )
    project_type: ProjectTypeEnum = Field(
        ..., description="Nature or type of services required in the project"
    )
    country: str = Field(
        ..., description="Primary country where the project takes place"
    )
    country_code: str = Field(
        ..., description="Standard 2-letter ISO country code in uppercase (e.g. PA, CO, ES, PE, MX, US)"
    )
    region: str = Field(
        ..., description="Geographic region (e.g. LATAM, Central America, South America, Europe, North America)"
    )
    city_province: Optional[str] = Field(
        default=None, description="Specific city, province, or department if mentioned"
    )
    budget_raw: Optional[str] = Field(
        default=None, description="Raw budget text from document"
    )
    budget_usd_estimate: Optional[float] = Field(
        default=None, description="Estimated budget normalized to USD as a numeric float value"
    )
    budget_scale: BudgetScaleEnum = Field(
        ..., description="Budget range bucket normalized to USD scale"
    )
    funding_source: FundingSourceEnum = Field(
        ..., description="Source of project financing (Public, Multilateral Bank, Private, PPP)"
    )
    multilateral_entity_name: Optional[str] = Field(
        default=None, description="Name of multilateral entity if applicable e.g. BID, Banco Mundial, CAF, BCIE"
    )
    execution_period_months: Optional[int] = Field(
        default=None, description="Duration of contract execution in months as an integer"
    )
    required_certifications: List[str] = Field(
        default_factory=list, description="List of required ISO or technical certifications"
    )
    consortium_allowed: Optional[bool] = Field(
        default=None, description="True if participation in consortium is permitted"
    )
    key_roles: List[str] = Field(
        default_factory=list, description="Top key personnel roles or specialist profiles required"
    )

    @field_validator("engineering_field", mode="before")
    @classmethod
    def coerce_engineering_field(cls, v: Any) -> Any:
        if isinstance(v, EngineeringFieldEnum):
            return v
        s = str(v).strip().lower()
        if "civil" in s or "infra" in s:
            return EngineeringFieldEnum.CIVIL_INFRASTRUCTURE
        if "water" in s or "sanitat" in s or "agua" in s or "saneam" in s:
            return EngineeringFieldEnum.WATER_SANITATION
        if "energ" in s or "power" in s or "eléctr" in s or "electr" in s:
            return EngineeringFieldEnum.ENERGY_POWER
        if "transport" in s or "road" in s or "vial" in s or "carretera" in s:
            return EngineeringFieldEnum.TRANSPORTATION_ROADS
        if "environ" in s or "ambient" in s or "natural" in s:
            return EngineeringFieldEnum.ENVIRONMENTAL
        if "build" in s or "facil" in s or "edific" in s or "archit" in s:
            return EngineeringFieldEnum.BUILDINGS_ARCHITECTURE
        if "telecom" in s or "it" in s or "tecnol" in s or "tech" in s:
            return EngineeringFieldEnum.TELECOM_IT
        if "industr" in s or "mining" in s or "minería" in s:
            return EngineeringFieldEnum.INDUSTRIAL_MINING
        for item in EngineeringFieldEnum:
            if item.value.lower() == s:
                return item
        return EngineeringFieldEnum.OTHER

    @field_validator("project_type", mode="before")
    @classmethod
    def coerce_project_type(cls, v: Any) -> Any:
        if isinstance(v, ProjectTypeEnum):
            return v
        s = str(v).strip().lower()
        if "supervis" in s or "inspect" in s or "fiscaliz" in s or "interventor" in s:
            return ProjectTypeEnum.CONSTRUCTION_SUPERVISION
        if "feasib" in s or "audit" in s or "design" in s or "diseño" in s or "estudio" in s or "factib" in s:
            return ProjectTypeEnum.FEASIBILITY_DESIGN
        if "maint" in s or "operat" in s or "manten" in s or "operac" in s:
            return ProjectTypeEnum.MAINTENANCE_OPERATION
        if "consult" in s or "asesor" in s or "estratég" in s:
            return ProjectTypeEnum.TECHNICAL_CONSULTING
        if "turnkey" in s or "epc" in s or "llave en mano" in s or "construc" in s:
            return ProjectTypeEnum.TURNKEY_EPC
        for item in ProjectTypeEnum:
            if item.value.lower() == s:
                return item
        return ProjectTypeEnum.OTHER

    @field_validator("budget_scale", mode="before")
    @classmethod
    def coerce_budget_scale(cls, v: Any) -> Any:
        if isinstance(v, BudgetScaleEnum):
            return v
        s = str(v).strip().lower()
        if "< 500" in s or "micro" in s or "under 500" in s:
            return BudgetScaleEnum.MICRO
        if "500k" in s and "2m" in s or "small" in s:
            return BudgetScaleEnum.SMALL
        if "2m" in s and "10m" in s or "medium" in s:
            return BudgetScaleEnum.MEDIUM
        if "10m" in s and "50m" in s or "large" in s:
            return BudgetScaleEnum.LARGE
        if "> 50m" in s or "mega" in s or "over 50m" in s:
            return BudgetScaleEnum.MEGA
        for item in BudgetScaleEnum:
            if item.value.lower() == s:
                return item
        return BudgetScaleEnum.UNDISCLOSED

    @field_validator("funding_source", mode="before")
    @classmethod
    def coerce_funding_source(cls, v: Any) -> Any:
        if isinstance(v, FundingSourceEnum):
            return v
        s = str(v).strip().lower()
        if "multilateral" in s or "bank" in s or "bid" in s or "world" in s or "caf" in s or "bcie" in s:
            return FundingSourceEnum.MULTILATERAL_BANK
        if "public" in s or "state" in s or "públic" in s or "estatal" in s or "gobierno" in s:
            return FundingSourceEnum.PUBLIC_STATE
        if "privat" in s or "privad" in s:
            return FundingSourceEnum.PRIVATE
        if "ppp" in s or "partner" in s or "app" in s or "mixt" in s:
            return FundingSourceEnum.MIXED
        for item in FundingSourceEnum:
            if item.value.lower() == s:
                return item
        return FundingSourceEnum.UNKNOWN
