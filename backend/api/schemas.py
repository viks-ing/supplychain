"""
Pydantic Schemas for Vyuha ML API.
Supports dynamic extra fields without hardcoded restrictions.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class CompanyProfile(BaseModel):
    company_name: Optional[str] = Field(default="Supply Chain Enterprise", description="Company Name")
    industry: Optional[str] = Field(default="Manufacturing", description="Industry Sector")
    monthly_procurement_lakhs: Optional[float] = Field(default=100.0, description="Monthly procurement value")
    
    class Config:
        extra = "allow"
        populate_by_name = True


class IndianConditions(BaseModel):
    shock_type: Optional[str] = Field(default="Normal Baseline", description="Disruption type")
    duration_days: Optional[float] = Field(default=0.0, description="Disruption duration in days")
    severity_pct: Optional[float] = Field(default=0.0, description="Severity percentage")
    
    class Config:
        extra = "allow"
        populate_by_name = True


class PredictRequest(BaseModel):
    company: CompanyProfile
    conditions: Optional[IndianConditions] = None

    class Config:
        extra = "allow"


class SimulateRequest(BaseModel):
    company: CompanyProfile
    shock_parameters: IndianConditions
    baseline_conditions: Optional[IndianConditions] = None

    class Config:
        extra = "allow"


class DataCoOrderRequest(BaseModel):
    """
    Dynamic DataCo order request schema.
    Accepts all standard Kaggle DataCo fields and allows arbitrary additional fields dynamically.
    """
    Type: Optional[str] = Field(default="DEBIT", description="Transaction type")
    Shipping_Mode: Optional[str] = Field(default="Standard Class", alias="Shipping Mode")
    Customer_Segment: Optional[str] = Field(default="Consumer", alias="Customer Segment")
    Market: Optional[str] = Field(default="LATAM", description="Market region")
    Order_Region: Optional[str] = Field(default="Central America", alias="Order Region")
    Department_Name: Optional[str] = Field(default="Apparel", alias="Department Name")
    Days_for_shipment_scheduled: Optional[int] = Field(default=4, alias="Days for shipment (scheduled)")
    Order_Item_Quantity: Optional[int] = Field(default=1, alias="Order Item Quantity")
    Product_Price: Optional[float] = Field(default=100.0, alias="Product Price")
    Order_Item_Discount: Optional[float] = Field(default=0.0, alias="Order Item Discount")
    Order_Item_Discount_Rate: Optional[float] = Field(default=0.0, alias="Order Item Discount Rate")
    Order_Item_Total: Optional[float] = Field(default=100.0, alias="Order Item Total")
    order_month: Optional[int] = Field(default=None)
    order_dayofweek: Optional[int] = Field(default=None)
    order_hour: Optional[int] = Field(default=None)

    class Config:
        extra = "allow"
        populate_by_name = True
